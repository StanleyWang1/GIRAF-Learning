"""Probe Luxonis DepthAI devices visible inside the container.

Usage (hardware machine):
    uv run --extra hardware python -m giraf.drivers.camera_probe
"""

import sys

import depthai as dai


def main():
    print(f"DepthAI version: {dai.__version__}")

    devices = dai.Device.getAllAvailableDevices()
    print(f"Visible DepthAI devices: {len(devices)}")

    if not devices:
        print("No DepthAI devices found.")
        print(
            "Check USB passthrough, cable/power, and whether the container was restarted after plugging in the camera."
        )
        return 1

    for idx, info in enumerate(devices):
        state = getattr(info, "state", "unknown")
        name = getattr(info, "name", "unknown")
        try:
            mxid = info.getMxId()
        except Exception:  # noqa: BLE001 - diagnostic probe reports and continues
            mxid = "unknown"
        print(f"[{idx}] name={name} mxid={mxid} state={state}")

    print("\nOpening first visible device for a basic handshake...")
    info = devices[0]

    try:
        with dai.Device(info) as device:
            print("Handshake succeeded.")
            print(f"Device MXID: {device.getMxId()}")
            print(f"USB speed: {device.getUsbSpeed()}")
            print(f"Connected cameras: {device.getConnectedCameras()}")
            print(f"Connected IMU: {device.getConnectedIMU()}")
    except Exception as exc:  # noqa: BLE001 - diagnostic probe reports and continues
        print(f"Failed to open DepthAI device: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
