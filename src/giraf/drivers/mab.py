"""MAB MD80 motor adapter."""

import time
import warnings

import pyCandle


def motor_connect():
    kp_small_motor = 100
    kd_small_motor = 5
    kp_large_motor = 1000
    kd_large_motor = 50
    max_torque = 25.0

    print("[INIT][MAB] Creating CANdle interface...", flush=True)
    candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
    print("[INIT][MAB] CANdle interface created.", flush=True)

    ids = {"ROLL": 21, "PITCH": 22, "BOOM": 23}
    for motor_id in ids.values():
        print(f"[INIT][MAB] Registering motor {motor_id}...", flush=True)
        candle.addMd80(motor_id)

    motors = {name: index for index, name in enumerate(ids)}
    for name, index in motors.items():
        motor = candle.md80s[index]
        print(f"[INIT][MAB] Configuring {name} motor (ID {ids[name]})...", flush=True)
        candle.controlMd80SetEncoderZero(motor)
        candle.controlMd80Mode(motor, pyCandle.IMPEDANCE)
        if ids[name] == 21:
            motor.setImpedanceControllerParams(kp_small_motor, kd_small_motor)
        else:
            motor.setImpedanceControllerParams(kp_large_motor, kd_large_motor)
        motor.setMaxTorque(max_torque)
        candle.controlMd80Enable(motor, True)
        print(f"[INIT][MAB] {name} motor ready.", flush=True)

    print("[INIT][MAB] Starting CANdle update loop...", flush=True)
    candle.begin()
    print("[INIT][MAB] CANdle update loop running.", flush=True)
    return candle, motors


def motor_status(candle, motors):
    error_flags = {
        0: "Main encoder error",
        1: "Output encoder error",
        2: "Calibration encoder error",
        3: "MOSFET bridge error",
        4: "Hardware error",
        5: "Communication error",
        6: "Motion error",
    }
    for name, index in motors.items():
        motor = candle.md80s[index]
        status = motor.getQuickStatus()
        for bit, message in error_flags.items():
            if status & (1 << bit):
                warnings.warn(
                    f"Motor {name} (ID {motor.getId()}) {message}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if status & (1 << 15):
            print(
                f"Motor {name} (ID {motor.getId()}) has reached its target "
                "position or velocity."
            )


def motor_drive(candle, motors, roll, pitch, boom):
    candle.md80s[motors["ROLL"]].setTargetPosition(roll)
    candle.md80s[motors["PITCH"]].setTargetPosition(pitch)
    candle.md80s[motors["BOOM"]].setTargetPosition(boom)


def motor_disconnect(candle):
    print("[STOP][MAB] Stopping CANdle...", flush=True)
    candle.end()
    print("[STOP][MAB] CANdle stopped.", flush=True)


def main():
    candle = None
    try:
        candle, motors = motor_connect()
        print("MAB motors connected; holding position 0 for 5 seconds.")
        motor_drive(candle, motors, 0.0, 0.0, 0.0)
        time.sleep(5.0)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if candle is not None:
            motor_disconnect(candle)
            print("MAB motors disabled and disconnected.")


if __name__ == "__main__":
    main()
