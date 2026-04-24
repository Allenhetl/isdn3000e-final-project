from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, Vector3
from sensor_msgs.msg import JointState

from ttt_interfaces.msg import PieceState, WorkspaceLayout


def _make_pose(x: float, y: float, z: float) -> Pose:
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


@dataclass(frozen=True)
class LayoutData:
    raw: dict[str, Any]
    workspace: WorkspaceLayout


def load_layout() -> LayoutData:
    share_dir = Path(get_package_share_directory("ttt_layout"))
    config_path = share_dir / "config" / "board_layout.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    workspace = WorkspaceLayout()
    workspace.planning_frame = str(raw["planning_frame"])
    workspace.ee_frame = str(raw["ee_frame"])
    workspace.attach_distance_threshold = float(raw["attach_distance_threshold"])
    workspace.pick_approach_height = float(raw["pick_approach_height"])
    workspace.place_approach_height = float(raw["place_approach_height"])

    home = JointState()
    home.name = list(raw["home_joint_state"]["names"])
    home.position = [float(value) for value in raw["home_joint_state"]["positions"]]
    workspace.home_joint_state = home

    size = Vector3()
    size.x = float(raw["piece_size"]["x"])
    size.y = float(raw["piece_size"]["y"])
    size.z = float(raw["piece_size"]["z"])
    workspace.piece_size = size

    cell_poses = [_make_pose(0.0, 0.0, 0.0) for _ in range(9)]
    for cell in raw["cells"]:
        cell_poses[int(cell["id"])] = _make_pose(cell["x"], cell["y"], cell["z"])
    workspace.cell_poses = cell_poses

    initial_pieces: list[PieceState] = []
    for piece in raw["pieces"]:
        state = PieceState()
        state.piece_id = int(piece["piece_id"])
        state.owner = int(piece["owner"])
        state.available = True
        state.location = PieceState.LOCATION_STOCK
        state.cell_id = 255
        state.pose = _make_pose(piece["x"], piece["y"], piece["z"])
        initial_pieces.append(state)
    workspace.initial_pieces = initial_pieces

    return LayoutData(raw=raw, workspace=workspace)
