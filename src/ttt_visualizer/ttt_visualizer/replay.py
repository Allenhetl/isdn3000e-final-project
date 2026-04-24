from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import Pose

from ttt_interfaces.msg import PieceState, TurnPlan


PHASE_NAMES = (
    "home_to_pick",
    "pick_to_home",
    "home_to_place",
    "place_to_home",
)

VIEW_MODES = (
    "motion",
    "committed_snapshot",
)


@dataclass(frozen=True)
class ReplaySelection:
    phase_name: str
    point_index: int


def clone_pose(pose: Pose) -> Pose:
    clone = Pose()
    clone.position.x = pose.position.x
    clone.position.y = pose.position.y
    clone.position.z = pose.position.z
    clone.orientation.x = pose.orientation.x
    clone.orientation.y = pose.orientation.y
    clone.orientation.z = pose.orientation.z
    clone.orientation.w = pose.orientation.w
    return clone


def clone_piece(piece: PieceState) -> PieceState:
    clone = PieceState()
    clone.piece_id = piece.piece_id
    clone.owner = piece.owner
    clone.available = piece.available
    clone.location = piece.location
    clone.cell_id = piece.cell_id
    clone.pose = clone_pose(piece.pose)
    return clone


def get_phase_trajectory(plan: TurnPlan, phase_name: str):
    return getattr(plan, phase_name).joint_trajectory


def get_phase_point_count(plan: TurnPlan, phase_name: str) -> int:
    return len(get_phase_trajectory(plan, phase_name).points)


def clamp_point_index(point_index: int, point_count: int) -> int:
    if point_count <= 0:
        return 0
    return max(0, min(int(point_index), point_count - 1))


def safe_point_index(
    value: float | int | None, point_count: int, fallback: int = 0
) -> int:
    if value is None:
        return clamp_point_index(fallback, point_count)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return clamp_point_index(fallback, point_count)
    if not np.isfinite(numeric_value):
        return clamp_point_index(fallback, point_count)
    return clamp_point_index(int(round(numeric_value)), point_count)


def motion_base_snapshot_turn(selected_turn: int) -> int:
    return int(selected_turn)


def committed_snapshot_turn(selected_turn: int) -> int:
    return int(selected_turn) + 1


def replay_piece_states(
    base_pieces: list[PieceState],
    plan: TurnPlan,
    target_pose: Pose,
    tcp_poses_by_phase: dict[str, list[Pose]],
    selection: ReplaySelection,
) -> dict[int, PieceState]:
    pieces = {piece.piece_id: clone_piece(piece) for piece in base_pieces}
    active_piece = pieces.get(int(plan.piece_id))
    if active_piece is None:
        return pieces

    if selection.phase_name in ("pick_to_home", "home_to_place"):
        tcp_poses = tcp_poses_by_phase.get(selection.phase_name, [])
        if tcp_poses:
            point_index = clamp_point_index(selection.point_index, len(tcp_poses))
            active_piece.pose = clone_pose(tcp_poses[point_index])
        active_piece.location = PieceState.LOCATION_ATTACHED
        active_piece.available = False
        active_piece.cell_id = 255
    elif selection.phase_name == "place_to_home":
        active_piece.location = PieceState.LOCATION_BOARD
        active_piece.available = False
        active_piece.cell_id = plan.cell_id
        active_piece.pose = clone_pose(target_pose)

    pieces[int(plan.piece_id)] = clone_piece(active_piece)
    return pieces
