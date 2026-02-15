"""
Analog Gauge Meter Reading - Main Application
===============================================
Supports three input modes:
  1. Image  — read a single image and display result
  2. Webcam — real-time gauge reading from webcam
  3. RTSP   — real-time gauge reading from RTSP stream

Usage:
  cd /home/ozzaann/gauge_model
  source gauge_meter_analog_reading_realtime/.gauge_meter/bin/activate
  python -m gauge_meter_analog_reading_realtime.app.main
"""

import sys
import os
import cv2
import yaml
import logging
import argparse
import time

# Ensure the project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gauge_meter_analog_reading_realtime.internal.ai_runtime.gauge_read import GaugeReader

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GaugeMeterApp")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from: {config_path}")
    return config


def run_image_mode(reader: GaugeReader, image_path: str, config: dict):
    """Read gauge from a single image and display the result."""
    logger.info(f"Image mode — reading: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Failed to read image: {image_path}")
        return

    readings = reader.read_gauge(frame)

    if not readings:
        logger.warning("No gauge detected in the image.")
    else:
        for i, reading in enumerate(readings):
            logger.info(
                f"Gauge #{i + 1}: {reading['value']:.2f} {reading['unit']} "
                f"(angle={reading['angle']:.1f}°, conf={reading['confidence']:.2f})"
            )

    # Draw results
    display_cfg = config.get("display", {})
    annotated = reader.draw_result(
        frame, readings,
        show_keypoints=display_cfg.get("show_keypoints", True),
        show_angle=display_cfg.get("show_angle", True),
        show_bbox=display_cfg.get("show_bbox", True),
    )

    # Resize for display
    win_w = display_cfg.get("window_width", 800)
    win_h = display_cfg.get("window_height", 600)
    display_frame = cv2.resize(annotated, (win_w, win_h))

    cv2.imshow("Analog Gauge Reader", display_frame)
    logger.info("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output
    output_dir = os.path.join(os.path.dirname(image_path), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gauge_reading_result.jpg")
    cv2.imwrite(output_path, annotated)
    logger.info(f"Result saved to: {output_path}")


def run_realtime_mode(reader: GaugeReader, source, config: dict, mode_name: str):
    """
    Real-time gauge reading from webcam or RTSP stream.

    Args:
        reader: GaugeReader instance
        source: Webcam ID (int) or RTSP URL (str)
        config: Full configuration dict
        mode_name: "webcam" or "rtsp" for logging
    """
    logger.info(f"{mode_name.upper()} mode — connecting to source: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open {mode_name} source: {source}")
        return

    logger.info(f"Connected to {mode_name}. Press 'q' to quit.")

    display_cfg = config.get("display", {})
    win_w = display_cfg.get("window_width", 800)
    win_h = display_cfg.get("window_height", 600)

    fps_start_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from {mode_name}.")
                time.sleep(0.1)
                continue

            # Run gauge reading
            readings = reader.read_gauge(frame)

            # Draw results
            annotated = reader.draw_result(
                frame, readings,
                show_keypoints=display_cfg.get("show_keypoints", True),
                show_angle=display_cfg.get("show_angle", True),
                show_bbox=display_cfg.get("show_bbox", True),
            )

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(
                    annotated, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

            # Log readings periodically
            if frame_count % 30 == 0 and readings:
                for reading in readings:
                    logger.info(
                        f"[{mode_name.upper()}] Reading: {reading['value']:.2f} {reading['unit']} "
                        f"(angle={reading['angle']:.1f}°)"
                    )

            # Resize and display
            display_frame = cv2.resize(annotated, (win_w, win_h))
            cv2.imshow("Analog Gauge Reader - Realtime", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("User pressed 'q'. Exiting...")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"{mode_name.upper()} mode finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Analog Gauge Meter Reading - Real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="gauge_meter_analog_reading_realtime/config/gauge_config.yaml",
        help="Path to gauge config YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "webcam", "rtsp"],
        default=None,
        help="Input mode (overrides config)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image path (overrides config source.path)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize GaugeReader
    model_cfg = config.get("model", {})
    gauge_cfg = config.get("gauge", {})

    reader = GaugeReader(
        model_path=model_cfg.get("path", "gauge_meter_analog_reading_realtime/models/gauge-pose.pt"),
        gauge_config=gauge_cfg,
        confidence=model_cfg.get("confidence", 0.5),
    )

    # Determine mode
    source_cfg = config.get("source", {})
    mode = args.mode or source_cfg.get("type", "image")

    if mode == "image":
        image_path = args.image or source_cfg.get("path", "test_image.png")
        run_image_mode(reader, image_path, config)

    elif mode == "webcam":
        webcam_id = source_cfg.get("webcam_id", 0)
        run_realtime_mode(reader, webcam_id, config, "webcam")

    elif mode == "rtsp":
        rtsp_url = source_cfg.get("rtsp_url", "")
        if not rtsp_url:
            logger.error("RTSP URL is not configured. Set 'source.rtsp_url' in config.")
            return
        run_realtime_mode(reader, rtsp_url, config, "rtsp")

    else:
        logger.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
