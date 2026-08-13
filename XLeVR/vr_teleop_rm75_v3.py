#!/usr/bin/env python3
"""
vr_teleop_rm75_v2.py — 메타퀘스트 오른손으로 RM-75 텔레옵 + 데이터 녹화 (v2)

컨트롤 스킴 (표준 방식)
  - 그립(손잡이 안쪽, 중지):  쥐고 있는 동안 팔이 손을 따라오고 + 그 구간이 녹화됨
                              놓으면 팔 정지 + 에피소드가 recordings/ 에 저장됨
  - 트리거(검지, 아날로그):   그리퍼 개폐 정도를 연속 제어 (살짝=조금 닫힘, 끝까지=완전 닫힘)
  - v2도 위치만 제어, 자세는 클러치 쥔 순간 값으로 고정

사전 조건: 이 스크립트를 쓰려면 XLeVR 쪽 패치 2개가 필요합니다 (대화 참고).
  패치가 안 되어 있으면 자동으로 v1 방식(트리거=클러치, 그리퍼 없음)으로 동작합니다.

사용 순서
  1) ~/XLeRobot/XLeVR/ 에 저장, vr_monitor.py는 종료된 상태에서 실행
  2) DRY_RUN=True 로 먼저: 그립 쥐고 이동 → 좌표 확인, 트리거 → 그리퍼 값 확인
  3) 그리퍼가 실제로 장착·배선되어 있으면 GRIPPER_ENABLED=True
  4) 확인 끝나면 DRY_RUN=False 로 실기동 (비상정지 근처, 서보 ON, 공간 확보)
"""

import asyncio
import json
import os
import threading
import time
from datetime import datetime

import numpy as np

from vr_monitor import VRMonitor  # 같은 폴더의 XLeVR 모니터

# ========================= 사용자 설정 =========================
DRY_RUN = False           # 첫 실행은 반드시 True
ROBOT_IP = "192.168.1.18"
RATE_HZ = 40
POS_SCALE = 1.0          # 손 1cm -> 로봇 1cm
MAX_STEP = 0.003         # 사이클당 최대 이동 [m] (40Hz 기준 약 12cm/s 상한)
GRIP_ON = 0.5            # 클러치 판정 임계값
TRIGGER_DEADBAND = 0.03  # 트리거 미세 잔떨림 무시

GRIPPER_ENABLED = False  # EG2-4C2가 팔 끝에 실제 장착·배선된 경우에만 True
GRIPPER_MIN_DIFF = 20    # 이만큼 변해야 명령 전송 (1~1000 스케일)
GRIPPER_MIN_DT = 0.08    # 그리퍼 명령 최소 간격 [s]

RECORD = True            # 그립 구간 녹화 저장 여부
RECORD_DIR = "recordings"

# 작업공간 제한: '시작 포즈' 기준 각 방향 허용 이동량 [m]
# (스크립트 시작 시 팔의 현재 위치를 중심으로 안전 박스를 자동 생성)
WS_FWD, WS_BACK = 0.25, 0.10     # 앞 +25cm / 뒤 -10cm
WS_LEFT, WS_RIGHT = 0.18, 0.18   # 좌우 각 18cm (전체 폭 36cm)
WS_UP, WS_DOWN = 0.15, 0.15     # 위 +15cm / 아래 -15cm


def vr_delta_to_robot(d):
    """WebXR(x=오른쪽, y=위, z=뒤) -> RM-75 베이스(x=앞, y=왼쪽, z=위).
    DRY_RUN에서 방향이 반대인 축은 부호만 바꾸세요."""
    return np.array([-d[2], -d[0], d[1]])
# ==============================================================


def get_vr_state(monitor):
    """(VR 위치[m], 트리거 0..1, 그립 bool|None) 반환. 그립 None = 패치 미적용."""
    goal = monitor.get_right_goal_nowait()
    if goal is None or goal.metadata is None:
        return None
    md = goal.metadata
    vp = md.get("vr_position")
    if vp is None:
        return None
    grip = md.get("grip_active", None)  # 패치 2가 없으면 None
    return np.array(vp, dtype=float), float(md.get("trigger", 0.0)), grip


class RM75:
    def __init__(self, ip):
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, 8080)
        if self.handle.id == -1:
            raise RuntimeError("RM-75 연결 실패 — 전원/이더넷/IP 확인")
        print(f"### [RM75] 연결됨 (handle {self.handle.id})")
        self.arm.rm_set_tool_voltage(3)  # 말단 24V 보장 (그리퍼 전원)

    def state(self):
        """(pose[6] m/rad, joint[7] deg) — 실패 시 (None, None)"""
        try:
            ret, s = self.arm.rm_get_current_arm_state()
            if ret != 0:
                return None, None
            return (np.array(s["pose"], dtype=float),
                    list(map(float, s.get("joint", []))))
        except Exception:
            return None, None

    def send(self, pose6):
        self.arm.rm_movep_canfd([float(v) for v in pose6], False)

    def gripper(self, pos_1_to_1000):
        # 비차단 전송: 제어 루프를 막지 않음
        self.arm.rm_set_gripper_position(int(pos_1_to_1000), False, 0)

    def stop(self):
        try:
            self.arm.rm_set_arm_slow_stop()
        except Exception:
            pass
        try:
            self.arm.rm_delete_robot_arm()
        except Exception:
            pass


def save_episode(rows):
    if not rows:
        return None
    os.makedirs(RECORD_DIR, exist_ok=True)
    path = os.path.join(
        RECORD_DIR, f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def main():
    print("### VR 서버 시작 중... (퀘스트 브라우저 https://<PC IP>:8443, 접속 중이면 새로고침)")
    monitor = VRMonitor()
    if not monitor.initialize():
        print("### VR 모니터 초기화 실패")
        return
    threading.Thread(
        target=lambda: asyncio.run(monitor.start_monitoring()), daemon=True
    ).start()

    robot = None
    if DRY_RUN:
        print("### [DRY RUN] 로봇/그리퍼에 명령을 보내지 않습니다.")
        target = np.array([0.40, 0.0, 0.30, 0.0, 0.0, 0.0])
    else:
        robot = RM75(ROBOT_IP)
        pose, _ = robot.state()
        if pose is None:
            print("### 팔 상태를 읽지 못했습니다. 종료.")
            robot.stop()
            return
        target = pose
        print(f"### [RM75] 현재 포즈: {np.round(target, 3)}")

    ws_min = target[:3] + np.array([-WS_BACK, -WS_RIGHT, -WS_DOWN])
    ws_max = target[:3] + np.array([WS_FWD, WS_LEFT, WS_UP])
    print(f"### 작업공간(시작점 기준): x[{ws_min[0]:.2f}~{ws_max[0]:.2f}] "
          f"y[{ws_min[1]:.2f}~{ws_max[1]:.2f}] z[{ws_min[2]:.2f}~{ws_max[2]:.2f}]")

    engaged = False
    warned_no_patch = False
    vr_ref = None
    pose_ref = target.copy()
    grip_val = 1000  # 그리퍼 현재 목표 (1000=완전 열림, 1=완전 닫힘)
    last_trigger = 0.0
    last_grip_sent, last_grip_t = grip_val, 0.0
    episode_rows = []
    dt = 1.0 / RATE_HZ
    last_print = 0.0

    print("### 준비 완료 — 그립(중지)을 쥔 동안: 팔 추종 + 녹화 / 검지 트리거: 그리퍼. Ctrl+C 종료.")
    try:
        while True:
            t0 = time.time()
            s = get_vr_state(monitor)
            if s is not None:
                vr_pos, trigger, grip = s
                last_trigger = trigger

                # ---- 클러치 소스 결정 (패치 미적용 시 v1 폴백) ----
                if grip is None:
                    if not warned_no_patch:
                        warned_no_patch = True
                        print("### [경고] grip_active가 안 넘어옵니다 — 패치 2 미적용."
                              " 트리거=클러치(v1) 모드로 폴백, 그리퍼 제어는 비활성화.")
                    clutch = trigger >= GRIP_ON
                    trigger_for_gripper = None
                else:
                    clutch = bool(grip)
                    trigger_for_gripper = trigger

                # ---- 팔 추종 ----
                if clutch:
                    if not engaged:
                        engaged = True
                        vr_ref = vr_pos.copy()
                        if robot is not None:
                            pose, _ = robot.state()
                            pose_ref = pose if pose is not None else target.copy()
                        else:
                            pose_ref = target.copy()
                        target = pose_ref.copy()
                        episode_rows = []
                        print("\n### 🔴 [클러치 ON] 녹화 시작 — 팔이 손을 따라갑니다.")
                    d_vr = (vr_pos - vr_ref) * POS_SCALE
                    goal_xyz = np.clip(pose_ref[:3] + vr_delta_to_robot(d_vr),
                                       ws_min, ws_max)
                    step = np.clip(goal_xyz - target[:3], -MAX_STEP, MAX_STEP)
                    target[:3] += step
                    if robot is not None:
                        robot.send(target)
                elif engaged:
                    engaged = False
                    print(f"\n### ⏹ [클러치 OFF] 녹화 종료 ({len(episode_rows)} 스텝) — 팔 정지.")
                    if RECORD:
                        path = save_episode(episode_rows)
                        if path:
                            print(f"### 💾 저장 완료: {path}")
                    episode_rows = []

                # ---- 그리퍼 (검지 트리거, 아날로그) ----
                if trigger_for_gripper is not None:
                    tg = 0.0 if trigger_for_gripper < TRIGGER_DEADBAND else trigger_for_gripper
                    grip_val = int(round(1000 - tg * 999))  # 0->1000(열림), 1->1(닫힘)
                    now = time.time()
                    if (GRIPPER_ENABLED and robot is not None
                            and abs(grip_val - last_grip_sent) >= GRIPPER_MIN_DIFF
                            and now - last_grip_t >= GRIPPER_MIN_DT):
                        robot.gripper(grip_val)
                        last_grip_sent, last_grip_t = grip_val, now

                # ---- 녹화 ----
                if engaged and RECORD:
                    pose_now, joint_now = (robot.state() if robot is not None
                                           else (None, None))
                    episode_rows.append({
                        "t": time.time(),
                        "target_pose": [round(float(v), 5) for v in target],
                        "pose": ([round(float(v), 5) for v in pose_now]
                                 if pose_now is not None else None),
                        "joint": joint_now,
                        "gripper": grip_val,
                        "trigger": round(float(trigger), 3),
                        "vr_pos": [round(float(v), 5) for v in vr_pos],
                    })

            now = time.time()
            interval = 0.2 if engaged else 1.0
            if now - last_print > interval:
                last_print = now
                if engaged:
                    print(f"### 🔴 rec={len(episode_rows):4d} | "
                          f"xyz={np.round(target[:3], 3)} | "
                          f"trig={last_trigger:.2f} grip={grip_val}")
                else:
                    print(f"### 대기 | xyz={np.round(target[:3], 3)} | grip={grip_val}")

            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n### 종료합니다.")
        if engaged and RECORD:
            path = save_episode(episode_rows)
            if path:
                print(f"### 마지막 에피소드 저장: {path}")
    finally:
        if robot is not None:
            robot.stop()
            print("### [RM75] 감속 정지 후 연결 해제 완료.")


if __name__ == "__main__":
    main()
