"""
Test Script for Gauge Reading
===============================
Tests the GaugeReader against the test_image.png
and validates the reading is approximately 6.4 kg/cm².

Usage:
  cd /home/ozzaann/gauge_model
  source gauge_meter_analog_reading_realtime/.gauge_meter/bin/activate
  python -m gauge_meter_analog_reading_realtime.test.test_gauge_read
"""

import sys
import os
import cv2
import yaml
import logging

# Ensure the project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gauge_meter_analog_reading_realtime.internal.ai_runtime.gauge_read import GaugeReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GaugeTest")

# === Test Parameters ===
EXPECTED_VALUE = 6.4  # kg/cm²
TOLERANCE = 1.5       # ± tolerance
CONFIG_PATH = "gauge_meter_analog_reading_realtime/config/gauge_config.yaml"
TEST_IMAGE_PATH = "test_image.png"


def run_test():
    logger.info("=" * 60)
    logger.info("  GAUGE READING TEST")
    logger.info("=" * 60)

    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    gauge_cfg = config.get("gauge", {})

    # Initialize reader
    reader = GaugeReader(
        model_path=model_cfg.get("path"),
        gauge_config=gauge_cfg,
        confidence=model_cfg.get("confidence", 0.5),
    )

    # Read test image
    frame = cv2.imread(TEST_IMAGE_PATH)
    if frame is None:
        logger.error(f"FAIL: Cannot read test image: {TEST_IMAGE_PATH}")
        return False

    logger.info(f"Image loaded: {TEST_IMAGE_PATH} ({frame.shape[1]}x{frame.shape[0]})")

    # Run gauge reading
    readings = reader.read_gauge(frame)

    if not readings:
        logger.error("FAIL: No gauge detected in the test image.")
        logger.info("Hint: Check model path and confidence threshold in config.")
        
        # Run raw detection for debugging
        logger.info("Running raw detection for diagnostics...")
        detections = reader.detect_gauge(frame)
        logger.info(f"Raw detections found: {len(detections)}")
        for i, det in enumerate(detections):
            logger.info(f"  Detection #{i}: bbox={det['bbox']}, conf={det['det_conf']:.3f}")
            for j, kp in enumerate(det['keypoints']):
                logger.info(f"    Keypoint {j}: x={kp[0]:.1f}, y={kp[1]:.1f}, conf={kp[2]:.3f}")
        return False

    # Report results
    logger.info(f"\nDetected {len(readings)} gauge(s):")
    all_passed = True
    for i, reading in enumerate(readings):
        value = reading["value"]
        angle = reading["angle"]
        unit = reading["unit"]
        conf = reading["confidence"]

        logger.info(f"\n  Gauge #{i + 1}:")
        logger.info(f"    Value     : {value:.2f} {unit}")
        logger.info(f"    Angle     : {angle:.2f}°")
        logger.info(f"    Confidence: {conf:.3f}")
        logger.info(f"    Center    : ({reading['center'][0]:.1f}, {reading['center'][1]:.1f})")
        logger.info(f"    Needle Tip: ({reading['needle_tip'][0]:.1f}, {reading['needle_tip'][1]:.1f})")

        # Validate
        error = abs(value - EXPECTED_VALUE)
        passed = error <= TOLERANCE

        if passed:
            logger.info(f"    ✅ PASS: Reading {value:.2f} within ±{TOLERANCE} of expected {EXPECTED_VALUE}")
        else:
            logger.error(f"    ❌ FAIL: Reading {value:.2f} is off by {error:.2f} from expected {EXPECTED_VALUE}")
            all_passed = False

    # Save annotated output
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    annotated = reader.draw_result(frame, readings)
    output_path = os.path.join(output_dir, "test_result.jpg")
    cv2.imwrite(output_path, annotated)
    logger.info(f"\nAnnotated result saved to: {output_path}")

    logger.info("=" * 60)
    if all_passed:
        logger.info("  TEST RESULT: ✅ ALL PASSED")
    else:
        logger.info("  TEST RESULT: ❌ SOME TESTS FAILED")
        logger.info("  Hint: Adjust min_angle/max_angle in gauge_config.yaml")
    logger.info("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
