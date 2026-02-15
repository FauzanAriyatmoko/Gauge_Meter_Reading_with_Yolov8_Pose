"""
Gauge Meter Analog Reading Module
==================================
Core logic for reading analog gauge meters using YOLOv8 Pose model.

The model detects keypoints on the gauge:
  - Keypoint 0: Center of the gauge
  - Keypoint 1: Tip of the needle

The angle between center and needle tip is computed and mapped
to a physical value based on configurable min/max angle and value.
"""

import cv2
import numpy as np
import math
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class GaugeReader:
    """
    Reads analog gauge meter values using a YOLOv8 Pose model.

    The model detects gauge keypoints (center + needle tip),
    and the class computes the needle angle to map it to a
    physical gauge reading.

    Args:
        model_path (str): Path to the YOLOv8 Pose model (.pt file)
        gauge_config (dict): Gauge calibration config with keys:
            - min_value (float): Minimum gauge reading
            - max_value (float): Maximum gauge reading  
            - min_angle (float): Angle in degrees at min_value position
            - max_angle (float): Angle in degrees at max_value position
            - unit (str): Unit of measurement (e.g., "kg/cm²")
        confidence (float): Detection confidence threshold
    """

    def __init__(self, model_path: str, gauge_config: dict, confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.gauge_config = gauge_config
        self.confidence = confidence

        # Extract gauge calibration parameters
        self.min_value = gauge_config.get("min_value", 0)
        self.max_value = gauge_config.get("max_value", 10)
        self.min_angle = gauge_config.get("min_angle", 225)
        self.max_angle = gauge_config.get("max_angle", -45)
        self.unit = gauge_config.get("unit", "kg/cm2")

        logger.info(
            f"GaugeReader initialized: "
            f"value=[{self.min_value}, {self.max_value}] {self.unit}, "
            f"angle=[{self.min_angle}°, {self.max_angle}°], "
            f"confidence={self.confidence}"
        )

    def detect_gauge(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 Pose inference on a frame to detect gauges and keypoints.

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of detection dicts, each containing:
              - bbox: [x1, y1, x2, y2]
              - keypoints: list of (x, y, conf) tuples
              - det_conf: detection confidence
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            keypoints_data = result.keypoints.data.cpu().numpy()

            for i in range(len(boxes)):
                kps = keypoints_data[i]  # Shape: (num_keypoints, 3) -> x, y, conf
                det = {
                    "bbox": boxes[i].tolist(),
                    "keypoints": [(float(kp[0]), float(kp[1]), float(kp[2])) for kp in kps],
                    "det_conf": float(confs[i]),
                }
                detections.append(det)

        return detections

    @staticmethod
    def compute_angle(center: tuple, needle_tip: tuple) -> float:
        """
        Compute the angle of the needle relative to the gauge center.

        Uses standard mathematical convention:
        - 0° = right (3 o'clock)
        - 90° = up (12 o'clock)
        - 180° = left (9 o'clock)
        - 270° / -90° = down (6 o'clock)

        Note: In image coordinates, y-axis is inverted (down is positive),
        so we negate dy to convert to standard math coordinates.

        Args:
            center: (x, y) of gauge center
            needle_tip: (x, y) of needle tip

        Returns:
            Angle in degrees [-180, 180]
        """
        dx = needle_tip[0] - center[0]
        dy = -(needle_tip[1] - center[1])  # Negate because y is inverted in image coords

        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def angle_to_value(self, angle: float) -> float:
        """
        Map a needle angle to a gauge value using linear interpolation.

        Handles clockwise sweeps (typical for analog gauges) where
        min_angle > max_angle in standard math convention.

        All angles are normalized to [0, 360) range before computing
        the sweep fraction to avoid wrap-around issues.

        Args:
            angle: Needle angle in degrees

        Returns:
            Gauge reading value
        """
        # Normalize all angles to [0, 360) range
        def normalize(a):
            return a % 360

        min_a = normalize(self.min_angle)
        max_a = normalize(self.max_angle)
        needle_a = normalize(angle)

        # Calculate the clockwise sweep from min to max
        # For a typical gauge: min_angle=220° (0 position) sweeping CW to max_angle=333° (max position)
        # The CW sweep passes through 0°, so sweep = (360 - min_a) + max_a
        # But if min_a < max_a (e.g., 30° to 330°), sweep = max_a - min_a (CCW)
        # We use the "clockwise distance from min to max" for typical gauges

        # Clockwise distance from min_a to max_a (going CW = decreasing angle in math convention)
        cw_sweep = (min_a - max_a) % 360

        # Clockwise distance from min_a to needle
        cw_needle = (min_a - needle_a) % 360

        # Fraction of sweep
        if cw_sweep < 0.001:
            return self.min_value

        fraction = cw_needle / cw_sweep
        fraction = max(0.0, min(1.0, fraction))

        # Linear interpolation
        value = self.min_value + fraction * (self.max_value - self.min_value)
        return value

    def read_gauge(self, frame: np.ndarray) -> list[dict]:
        """
        Full pipeline: detect gauge(s), compute angle, and map to value.

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of reading dicts, each containing:
              - value (float): The gauge reading
              - angle (float): The computed needle angle
              - unit (str): The unit of measurement
              - bbox (list): Bounding box [x1, y1, x2, y2]
              - center (tuple): (x, y) of gauge center
              - needle_tip (tuple): (x, y) of needle tip
              - confidence (float): Detection confidence
        """
        detections = self.detect_gauge(frame)
        readings = []

        for det in detections:
            kps = det["keypoints"]

            if len(kps) < 2:
                logger.warning("Detection has fewer than 2 keypoints, skipping")
                continue

            # Keypoint 0: Center, Keypoint 1: Needle tip
            center = (kps[0][0], kps[0][1])
            needle_tip = (kps[1][0], kps[1][1])
            kp_center_conf = kps[0][2]
            kp_needle_conf = kps[1][2]

            # Skip if keypoint confidence is too low
            if kp_center_conf < 0.3 or kp_needle_conf < 0.3:
                logger.warning(
                    f"Low keypoint confidence: center={kp_center_conf:.2f}, "
                    f"needle={kp_needle_conf:.2f}, skipping"
                )
                continue

            # Compute angle and value
            angle = self.compute_angle(center, needle_tip)
            value = self.angle_to_value(angle)

            readings.append({
                "value": round(value, 3),
                "angle": round(angle, 2),
                "unit": self.unit,
                "bbox": det["bbox"],
                "center": center,
                "needle_tip": needle_tip,
                "confidence": det["det_conf"],
                "kp_center_conf": kp_center_conf,
                "kp_needle_conf": kp_needle_conf,
            })

        return readings

    def draw_result(self, frame: np.ndarray, readings: list[dict],
                    show_keypoints: bool = True,
                    show_angle: bool = True,
                    show_bbox: bool = True) -> np.ndarray:
        """
        Annotate a frame with gauge reading results.

        Args:
            frame: BGR image to draw on (will be copied)
            readings: List of reading dicts from read_gauge()
            show_keypoints: Draw center and needle tip points
            show_angle: Display angle information
            show_bbox: Draw bounding box

        Returns:
            Annotated frame (copy)
        """
        annotated = frame.copy()

        for reading in readings:
            bbox = reading["bbox"]
            center = (int(reading["center"][0]), int(reading["center"][1]))
            needle_tip = (int(reading["needle_tip"][0]), int(reading["needle_tip"][1]))
            value = reading["value"]
            angle = reading["angle"]
            unit = reading["unit"]
            conf = reading["confidence"]

            x1, y1, x2, y2 = map(int, bbox)

            # Draw bounding box
            if show_bbox:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw keypoints and needle line
            if show_keypoints:
                # Center point (blue)
                cv2.circle(annotated, center, 6, (255, 0, 0), -1)
                cv2.circle(annotated, center, 8, (255, 255, 255), 2)

                # Needle tip (red)
                cv2.circle(annotated, needle_tip, 6, (0, 0, 255), -1)
                cv2.circle(annotated, needle_tip, 8, (255, 255, 255), 2)

                # Needle line (cyan)
                cv2.line(annotated, center, needle_tip, (255, 255, 0), 2)

            # Draw reading text
            # Main value text with background
            value_text = f"{value:.3f} {unit}"
            text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = x1
            text_y = y1 - 15

            # Background rectangle for text
            cv2.rectangle(
                annotated,
                (text_x - 5, text_y - text_size[1] - 10),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0), -1
            )
            cv2.putText(
                annotated, value_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA
            )

            # Angle and confidence info
            if show_angle:
                info_text = f"Angle: {angle:.1f} | Conf: {conf:.2f}"
                info_y = y2 + 25
                cv2.putText(
                    annotated, info_text,
                    (x1, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
                )

        return annotated
