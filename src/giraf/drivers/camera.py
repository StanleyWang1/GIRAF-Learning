"""DepthAI color-camera preview."""

import cv2
import depthai as dai

FRAME_SIZE = (640, 480)
FRAME_RATE = 30
WINDOW_NAME = "GIRAF Camera"


def camera_connect(
    frame_size=FRAME_SIZE,
    frame_rate=FRAME_RATE,
    *,
    imu_config=None,
):
    """Start a DepthAI color-camera pipeline and return it with its frame queue."""
    pipeline = dai.Pipeline()
    try:
        camera = pipeline.create(dai.node.Camera).build()
        rgb_output = camera.requestOutput(
            frame_size,
            type=dai.ImgFrame.Type.BGR888p,
            fps=frame_rate,
        )
        frame_queue = rgb_output.createOutputQueue()
        if imu_config is not None:
            imu_queue, metadata = _connect_imu(pipeline, imu_config)
        pipeline.start()
        if imu_config is not None:
            return pipeline, frame_queue, imu_queue, metadata
        return pipeline, frame_queue
    except BaseException:
        pipeline.stop()
        raise


def _connect_imu(pipeline, config):
    """Configure the requested reports without changing device calibration."""
    import uuid

    import numpy as np

    from giraf.data.imu import IMU_REPORTS

    metadata = {"enabled": config.enabled, "session_id": uuid.uuid4().hex}
    if not config.enabled:
        return None, metadata
    device = pipeline.getDefaultDevice()
    imu_type = device.getConnectedIMU()
    if imu_type != "BNO086":
        raise RuntimeError(f"IMU collection requires BNO086; connected: {imu_type!r}")
    calibration = device.readCalibration().eepromToJson()
    parameters = calibration.get("imuCalibrationParams", {})
    for name in ("accelerometer", "gyroscope"):
        matrix = np.asarray(parameters.get(name, []), dtype=float)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            raise RuntimeError(f"missing/invalid {name} IMU calibration parameters")
    rotation = np.asarray(
        calibration.get("imuExtrinsics", {}).get("rotationMatrix", []), dtype=float
    )
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)
    ):
        raise RuntimeError("missing/invalid IMU extrinsic rotation")
    metadata.update(
        {
            "device_id": device.getDeviceId(),
            "device_product": calibration.get("productName"),
            "imu_type": imu_type,
            "imu_firmware": str(device.getIMUFirmwareVersion()),
            "depthai_version": dai.__version__,
            "usb_speed": str(device.getUsbSpeed()),
            "requested_rate_hz": config.report_rate_hz,
            "reports": list(IMU_REPORTS),
            "imu_calibration": parameters,
            "imu_extrinsics": calibration["imuExtrinsics"],
            "vector_frame": "DepthAI IMU frame, SDK extrinsics/calibration applied",
            "quaternion_order": "xyzw = i,j,k,real",
            "quaternion_frame": "BNO086 native fused game rotation output; no host transform",
            "orientation_reference": "gravity-referenced tilt; arbitrary heading, yaw may drift",
            "physical_axis_validation": "not performed automatically",
        }
    )
    imu = pipeline.create(dai.node.IMU)
    for report in IMU_REPORTS:
        imu.enableIMUSensor(getattr(dai.IMUSensor, report), config.report_rate_hz)
    imu.setBatchReportThreshold(config.batch_report_threshold)
    imu.setMaxBatchReports(config.max_batch_reports)
    queue = imu.out.createOutputQueue(maxSize=config.queue_size, blocking=False)
    return queue, metadata


def camera_read(frame_queue):
    """Wait for and return the next OpenCV color frame."""
    return camera_read_message(frame_queue).getCvFrame()


def camera_read_message(frame_queue):
    """Wait for the next frame message, preserving its capture metadata."""
    return frame_queue.get()


def camera_disconnect(pipeline):
    """Stop the DepthAI camera pipeline."""
    if pipeline.isRunning():
        pipeline.stop()


def main():
    pipeline, frame_queue = camera_connect()

    try:
        while pipeline.isRunning():
            frame = camera_read(frame_queue)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera_disconnect(pipeline)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
