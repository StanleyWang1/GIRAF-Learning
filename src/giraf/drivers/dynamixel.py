import time

import numpy as np
from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite

from .dynamixel_config import (
    BAUDRATE,
    GOAL_POSITION,
    GRIPPER_ID,
    MOTOR21_HOME,
    MOTOR22_HOME,
    MOTOR23_HOME,
    MOTOR24_OPEN,
    OPERATING_MODE,
    PORT,
    TICKS_PER_REV,
    TORQUE_ENABLE,
    WRIST_IDS,
)
from .dynamixel_controller import DynamixelController

# Motor IDs
# BOOM_ENCODER = 10
JOINT1, JOINT2, JOINT3 = WRIST_IDS
GRIPPER = GRIPPER_ID


def dynamixel_connect():
    # Initialize controller
    controller = DynamixelController(PORT, BAUDRATE, 2.0)
    group_sync_write = GroupSyncWrite(
        controller.port_handler,
        controller.packet_handler,
        GOAL_POSITION[0],
        GOAL_POSITION[1],
    )

    # --------------------------------------------------
    # Reboot WRIST motors to ensure clean startup
    for motor_id in [JOINT1, JOINT2, JOINT3, GRIPPER]:
        dxl_comm_result, dxl_error = controller.packet_handler.reboot(
            controller.port_handler, motor_id
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(
                f"Failed to reboot Motor {motor_id}: {controller.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
        elif dxl_error != 0:
            print(
                f"Error rebooting Motor {motor_id}: {controller.packet_handler.getRxPacketError(dxl_error)}"
            )
        else:
            print(f"Motor {motor_id} rebooted successfully.")

    # Give motors time to reboot
    time.sleep(2)

    # --------------------------------------------------
    # Setup BOOM_ENCODER motor (read-only, passive mode)
    # dxl_comm_result, dxl_error = controller.packet_handler.reboot(controller.port_handler, BOOM_ENCODER)
    # if dxl_comm_result != COMM_SUCCESS:
    #     print(f"Failed to reboot Boom Encoder (Motor {BOOM_ENCODER}): {controller.packet_handler.getTxRxResult(dxl_comm_result)}")
    # elif dxl_error != 0:
    #     print(f"Error rebooting Boom Encoder (Motor {BOOM_ENCODER}): {controller.packet_handler.getRxPacketError(dxl_error)}")
    # else:
    #     print(f"Boom Encoder (Motor {BOOM_ENCODER}) rebooted successfully.")

    # time.sleep(1)  # Allow reboot
    # controller.WRITE(BOOM_ENCODER, OPERATING_MODE, 4)  # Extended position
    # controller.WRITE(BOOM_ENCODER, TORQUE_ENABLE, 0)   # Ensure motor is torque-off (read-only)

    # Set Control Mode
    controller.WRITE(JOINT1, OPERATING_MODE, 4)  # extended position control
    controller.WRITE(JOINT2, OPERATING_MODE, 4)
    controller.WRITE(JOINT3, OPERATING_MODE, 4)
    controller.WRITE(GRIPPER, OPERATING_MODE, 4)

    # Optional: Force Limit on Gripper
    # controller.WRITE(GRIPPER, PWM_LIMIT, 250)

    # Torque Enable
    controller.WRITE(JOINT1, TORQUE_ENABLE, 1)
    controller.WRITE(JOINT2, TORQUE_ENABLE, 1)
    controller.WRITE(JOINT3, TORQUE_ENABLE, 1)
    controller.WRITE(GRIPPER, TORQUE_ENABLE, 0)  # Gripper torque off for now

    return controller, group_sync_write


def dynamixel_drive(controller, group_sync_write, ticks):
    param_success = group_sync_write.addParam(
        JOINT1, ticks[0].to_bytes(4, "little", signed=True)
    )
    param_success &= group_sync_write.addParam(
        JOINT2, ticks[1].to_bytes(4, "little", signed=True)
    )
    param_success &= group_sync_write.addParam(
        JOINT3, ticks[2].to_bytes(4, "little", signed=True)
    )
    param_success &= group_sync_write.addParam(
        GRIPPER, ticks[3].to_bytes(4, "little", signed=True)
    )

    if not param_success:
        print("Failed to add parameters for SyncWrite")
        return False

    dxl_comm_result = group_sync_write.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(
            f"SyncWrite communication error: {controller.packet_handler.getTxRxResult(dxl_comm_result)}"
        )
        return False

    group_sync_write.clearParam()
    return True


# def dynamixel_boom_ticks(controller):
#     """Reads the current position (ticks) of the passive boom encoder motor (ID 10),
#     and converts it to a signed 32-bit integer if necessary."""
#     motor10_pos = controller.READ(BOOM_ENCODER, PRESENT_POSITION)
#     if motor10_pos is False:
#         print("Failed to read boom encoder position.")
#         return None
#     # Convert from unsigned to signed 32-bit integer
#     if motor10_pos >= 2**31:
#         motor10_pos -= 2**32
#     return motor10_pos

# def dynamixel_boom_meters(controller, homing_offset = 0):
#     """Calculate the current position (m) of the boom"""
#     # Encoder ticks to boom length [m]
#     slope = -4.21212481e-05 # boom length per tick
#     # y_intercept = 5.32822431e-02  # boom length at 0 ticks
#     y_intercept = 0.381
#     # Get encoder ticks
#     motor10_pos = dynamixel_boom_ticks(controller)
#     # Convert to boom position
#     boom_pos = slope * (motor10_pos - homing_offset) + y_intercept # [m]
#     return boom_pos


def dynamixel_disconnect(controller):
    # Torque OFF all motors individually (simple)
    controller.WRITE(JOINT1, TORQUE_ENABLE, 0)
    controller.WRITE(JOINT2, TORQUE_ENABLE, 0)
    controller.WRITE(JOINT3, TORQUE_ENABLE, 0)
    controller.WRITE(GRIPPER, TORQUE_ENABLE, 0)


def radians_to_ticks(rad):
    return int(rad / (2 * np.pi) * TICKS_PER_REV)


def main():
    controller, group_sync_write = dynamixel_connect()
    print("\033[93mDYNAMIXEL: Motors Connected, Driving to Home (5 sec)\033[0m")
    dynamixel_drive(
        controller,
        group_sync_write,
        [MOTOR21_HOME, MOTOR22_HOME, MOTOR23_HOME, MOTOR24_OPEN],
    )
    time.sleep(5)
    dynamixel_disconnect(controller)
    print("\033[93mDYNAMIXEL: Motors Disconnected, Torque Off\033[0m")


if __name__ == "__main__":
    main()
