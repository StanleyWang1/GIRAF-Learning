"""DepthAI color-camera preview."""

import cv2
import depthai as dai

FRAME_SIZE = (640, 480)
FRAME_RATE = 30
WINDOW_NAME = "GIRAF Camera"


def camera_connect(
    frame_size=FRAME_SIZE,
    frame_rate=FRAME_RATE,
):
    """Start a DepthAI color-camera pipeline and return it with its frame queue."""
    pipeline = dai.Pipeline()
    camera = pipeline.create(dai.node.Camera).build()
    rgb_output = camera.requestOutput(
        frame_size,
        type=dai.ImgFrame.Type.BGR888p,
        fps=frame_rate,
    )
    frame_queue = rgb_output.createOutputQueue()
    pipeline.start()
    return pipeline, frame_queue


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
