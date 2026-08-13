#!/usr/bin/env python3
"""
vr_teleop_rm75.py — 메타퀘스트 오른손 컨트롤러로 RM-75 텔레옵 (v1)

동작 방식
  - 오른손 '트리거'를 당기고 있는 동안만 팔이 손을 따라옵니다 (클러치/데드맨).
  - 트리거를 놓으면 그 자리에 정지. 다시 쥐면 현재 위치 기준으로 재시작(점프 없음).
  - v1은 위치(x,y,z)만 제어하고, 자세는 클러치를 쥔 순간의 값으로 고정합니다.

사용 순서
  1) 이 파일을 ~/XLeRobot/XLeVR/ 에 저장 (vr_monitor.py 옆이어야 import 됨)
  2) vr_monitor.py가 떠 있으면 종료할 것 — 이 스크립트가 VR 서버(8443)를 직접 띄움
  3) (rm75_teleop) 활성화 후:  python vr_teleop_rm75.py
  4) 퀘스트 브라우저에서 https://<PC IP>:8443 접속(이미 열려 있으면 새로고침) → VR 진입
  5) DRY_RUN=True 상태로 트리거를 쥐고 손을 앞/왼/위로 움직여
     터미널의 "### 목표 xyz"가 각각 x+/y+/z+ 로 증가하는지 확인
     → 방향이 다르면 아래 vr_delta_to_robot()의 부호를 수정
  6) 방향이 맞으면 DRY_RUN=False 로 바꾸고 실제 구동
     (비상정지 버튼 손 닿는 곳에, 팔 서보 ON 상태, 주변 공간 확보)
"""

import asyncio
import threading
import time

import numpy as np

from vr_monitor import VRMonitor  # 같은 폴더의 XLeVR 모니터

# ========================= 사용자 설정 =========================
DRY_RUN = True          # True: 로봇에 안 보내고 목표 좌표만 출력 (첫 실행은 반드시 True)
ROBOT_IP = "192.168.1.18"
RATE_HZ = 40            # 제어 주기
POS_SCALE = 1.0         # 손 1cm -> 로봇 1cm (익숙해지면 0.5~1.5로 조절)
MAX_STEP = 0.003        # 사이클당 최대 이동 [m] → 40Hz 기준 최대 약 12cm/s
TRIGGER_ON = 0.5        # 트리거 클러치 임계값

# 작업공간 제한 [m], 로봇 베이스 좌표계 기준. 책상/벽에 맞게 꼭 조정하세요.
WS_MIN = np.array([0.20, -0.40, 0.10])
WS_MAX = np.array([0.70, 0.40, 0.60])


def vr_delta_to_robot(d):
    """VR 좌표 변화량 -> 로봇 베이스 좌표 변화량.

    WebXR 프레임: x=오른쪽, y=위, z=뒤(사용자 쪽)
    RM-75 베이스: x=앞, y=왼쪽, z=위 로 가정.
    DRY_RUN에서 방향이 반대로 나오면 해당 성분의 부호만 바꾸면 됩니다.
    """
    return np.array([-d[2], -d[0], d[1]])
# ==============================================================


def get_vr_state(monitor):
    """오른손 컨트롤러의 (원시 VR 위치[m], 트리거값) 반환. 데이터 없으면 None."""
    goal = monitor.get_right_goal_nowait()
    if goal is None or goal.metadata is None:
        return None
    md = goal.metadata
    vp = md.get("vr_position")
    if vp is None:  # 트리거 이벤트 등 위치가 없는 goal은 건너뜀
        return None
    return np.array(vp, dtype=float), float(md.get("trigger", 0.0))


class RM75:
    def __init__(self, ip):
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, 8080)
        if self.handle.id == -1:
            raise RuntimeError("RM-75 연결 실패 — 전원/이더넷/IP 확인")
        print(f"### [RM75] 연결됨 (handle {self.handle.id})")

    def pose(self):
        ret, state = self.arm.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"팔 상태 읽기 실패 ret={ret}")
        return np.array(state["pose"], dtype=float)  # [x,y,z,rx,ry,rz] (m, rad)

    def send(self, pose6):
        # follow=False: 컨트롤러가 자체 보간하는 부드러운 추종 모드
        self.arm.rm_movep_canfd([float(v) for v in pose6], False)

    def stop(self):
        try:
            self.arm.rm_set_arm_slow_stop()
        except Exception:
            pass
        try:
            self.arm.rm_delete_robot_arm()
        except Exception:
            pass


def main():
    print("### VR 서버 시작 중... (퀘스트 브라우저로 https://<PC IP>:8443 접속)")
    monitor = VRMonitor()
    if not monitor.initialize():
        print("### VR 모니터 초기화 실패")
        return
    threading.Thread(
        target=lambda: asyncio.run(monitor.start_monitoring()), daemon=True
    ).start()

    robot = None
    if DRY_RUN:
        print("### [DRY RUN] 로봇에 명령을 보내지 않습니다. 좌표 검증 모드.")
        target = np.array([0.40, 0.0, 0.30, 0.0, 0.0, 0.0])  # 가상 시작 포즈
    else:
        robot = RM75(ROBOT_IP)
        target = robot.pose()
        print(f"### [RM75] 현재 포즈: {np.round(target, 3)}")

    engaged = False
    vr_ref = None
    pose_ref = target.copy()
    dt = 1.0 / RATE_HZ
    last_print = 0.0

    print("### 준비 완료. 오른손 트리거를 쥔 동안 팔이 손을 따라옵니다. Ctrl+C 종료.")
    try:
        while True:
            t0 = time.time()
            s = get_vr_state(monitor)
            if s is not None:
                vr_pos, trig = s
                if trig >= TRIGGER_ON:
                    if not engaged:
                        engaged = True
                        vr_ref = vr_pos.copy()
                        pose_ref = robot.pose() if robot is not None else target.copy()
                        target = pose_ref.copy()
                        print("\n### [클러치 ON] 지금부터 손을 따라갑니다.")
                    d_vr = (vr_pos - vr_ref) * POS_SCALE
                    goal_xyz = pose_ref[:3] + vr_delta_to_robot(d_vr)
                    goal_xyz = np.clip(goal_xyz, WS_MIN, WS_MAX)
                    step = np.clip(goal_xyz - target[:3], -MAX_STEP, MAX_STEP)
                    target[:3] += step
                    if robot is not None:
                        robot.send(target)
                elif engaged:
                    engaged = False
                    print("\n### [클러치 OFF] 정지 — 현재 위치 유지.")

            now = time.time()
            if now - last_print > 1.0:
                last_print = now
                tag = "ON " if engaged else "off"
                print(f"### 클러치[{tag}] 목표 xyz = {np.round(target[:3], 3)}")

            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n### 종료합니다.")
    finally:
        if robot is not None:
            robot.stop()
            print("### [RM75] 감속 정지 후 연결 해제 완료.")


if __name__ == "__main__":
    main()
