"""Synthetic ground-truth generator: calibrated pinhole camera, textured road, 3-D vehicles."""

from .render import FrameAnnotation, Renderer
from .scene import (
    CameraSpec,
    LightingSpec,
    RoadSpec,
    ScenarioConfig,
    SimVehicle,
    TrafficSpec,
    ZoneSpec,
    build_camera,
    generate_traffic,
)

__all__ = [
    "CameraSpec",
    "FrameAnnotation",
    "LightingSpec",
    "Renderer",
    "RoadSpec",
    "ScenarioConfig",
    "SimVehicle",
    "TrafficSpec",
    "ZoneSpec",
    "build_camera",
    "generate_traffic",
]
