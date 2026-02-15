"""
Bridge module for importing from 'gauge-pose' directory.
The directory name contains a hyphen which is not valid
for direct Python imports. This module re-exports GaugeReader.
"""

import importlib
import os
import sys

# Add the gauge-pose directory to the path dynamically
_gauge_pose_dir = os.path.join(
    os.path.dirname(__file__), "gauge-pose"
)

if _gauge_pose_dir not in sys.path:
    sys.path.insert(0, _gauge_pose_dir)

from gauge_read import GaugeReader

__all__ = ["GaugeReader"]
