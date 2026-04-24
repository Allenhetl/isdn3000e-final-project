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


def pose_to_xyz(pose: Pose) -> np.ndarray:
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)


def pose_from_xyz(position: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.w = 1.0
    return pose


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
    attach_distance_threshold: float,
    tcp_positions_by_phase: dict[str, list[np.ndarray]],
    selection: ReplaySelection,
) -> dict[int, PieceState]:
    pieces = {piece.piece_id: clone_piece(piece) for piece in base_pieces}
    active_piece = pieces.get(int(plan.piece_id))
    if active_piece is None:
        return pieces

    attached = False
    target_position = pose_to_xyz(target_pose)
    selected_phase_index = PHASE_NAMES.index(selection.phase_name)

    for phase_index, phase_name in enumerate(PHASE_NAMES):
        tcp_positions = tcp_positions_by_phase.get(phase_name, [])
        if phase_index > selected_phase_index:
            break

        last_point_index = len(tcp_positions) - 1
        if phase_index == selected_phase_index:
            last_point_index = clamp_point_index(selection.point_index, len(tcp_positions))

        if last_point_index < 0:
            continue

        for point_index in range(last_point_index + 1):
            tcp_position = tcp_positions[point_index]
            piece_position = pose_to_xyz(active_piece.pose)

            if (
                not attached
                and np.linalg.norm(tcp_position - piece_position)
                <= attach_distance_threshold
            ):
                attached = True
                active_piece.location = PieceState.LOCATION_ATTACHED
                active_piece.available = False

            if attached:
                active_piece.pose = pose_from_xyz(tcp_position)

            if (
                attached
                and phase_index >= 2
                and np.linalg.norm(tcp_position - target_position)
                <= attach_distance_threshold
            ):
                attached = False
                active_piece.location = PieceState.LOCATION_BOARD
                active_piece.cell_id = plan.cell_id
                active_piece.pose = clone_pose(target_pose)

    pieces[int(plan.piece_id)] = clone_piece(active_piece)
    return pieces
