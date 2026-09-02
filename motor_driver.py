import pyCandle
import warnings
import time

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

    motors = {name: i for i, name in enumerate(ids.keys())}
    
    for name, i in motors.items():
      md = candle.md80s[i]
      print(f"[INIT][MAB] Configuring {name} motor (ID {ids[name]})...", flush=True)

      candle.controlMd80SetEncoderZero(md)
      candle.controlMd80Mode(md, pyCandle.IMPEDANCE)

      if ids[name] == 21: # smaller motor used on base roll joint!
          md.setImpedanceControllerParams(kp_small_motor, kd_small_motor)
      else:
          md.setImpedanceControllerParams(kp_large_motor, kd_large_motor)

      md.setMaxTorque(max_torque)

      # Enable only after configuring mode, gains, and torque limit.
      candle.controlMd80Enable(md, True)
      print(f"[INIT][MAB] {name} motor ready.", flush=True)

    print("[INIT][MAB] Starting CANdle update loop...", flush=True)
    candle.begin()
    print("[INIT][MAB] CANdle update loop running.", flush=True)
    
    return candle, motors

def motor_status(candle, motors):
    error_flags = {
        0: "Main encoder error",
        0: "Main encoder error",
        1: "Output encoder error",
        2: "Calibration encoder error",
        3: "MOSFET bridge error",
        4: "Hardware error",
        5: "Communication error",
        6: "Motion error"
    }

    for name, index in motors.items():
        motor = candle.md80s[index]
        status = motor.getQuickStatus()
        for bit, message in error_flags.items():
            if status & (1 << bit):
                warnings.warn(f"Motor {name} (ID {motor.getId()}) {message}.", RuntimeWarning)
        if status & (1 << 15):
            print(f"Motor {name} (ID {motor.getId()}) has reached its target position or velocity.")

def motor_drive(candle, motors, roll, pitch, boom):
    candle.md80s[motors["ROLL"]].setTargetPosition(roll)
    candle.md80s[motors["PITCH"]].setTargetPosition(pitch)
    candle.md80s[motors["BOOM"]].setTargetPosition(boom)
    
    # print("Boom Torque: " + str(round(candle.md80s[motors["BOOM"]].getTorque(), 3)))

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

        # candle.begin() runs the update loop in the background.
        time.sleep(5.0)

    except KeyboardInterrupt:
        print("Interrupted.")

    finally:
        if candle is not None:
            motor_disconnect(candle)
            print("MAB motors disabled and disconnected.")


if __name__ == "__main__":
    main()
