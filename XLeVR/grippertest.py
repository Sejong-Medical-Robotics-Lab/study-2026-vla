import time
from Robotic_Arm.rm_robot_interface import *

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
h = arm.rm_create_robot_arm("192.168.1.18", 8080)

print("전원 리셋:", arm.rm_set_tool_voltage(0)); time.sleep(2)
print("24V 재인가:", arm.rm_set_tool_voltage(3)); time.sleep(3)  # 그리퍼 부팅 대기

for pos in (100, 1000, 100, 1000):      # 거의 닫기 <-> 활짝 열기, 2회 왕복
    print("->", pos, arm.rm_set_gripper_position(pos, False, 0))
    time.sleep(2.5)

print("최종 상태:", arm.rm_get_gripper_state())
arm.rm_delete_robot_arm()
