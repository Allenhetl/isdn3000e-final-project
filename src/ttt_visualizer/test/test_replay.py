from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Pose

from ttt_interfaces.msg import PieceState, TurnPlan
from ttt_visualizer.replay import (
    PHASE_NAMES,
    ReplaySelection,
    clamp_point_index,
    committed_snapshot_turn,
    motion_base_snapshot_turn,
    replay_piece_states,
    safe_point_index,
)


def _pose(x: float, y: float, z: float) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.w = 1.0
    return pose


def _piece(piece_id: int, owner: int, x: float, y: float, z: float) -> PieceState:
    piece = PieceState()
    piece.piece_id = piece_id
    piece.owner = owner
    piece.available = True
    piece.location = PieceState.LOCATION_STOCK
    piece.cell_id = 255
    piece.pose = _pose(x, y, z)
    return piece


def test_clamp_point_index_handles_bounds() -> None:
    assert clamp_point_index(-1, 3) == 0
    assert clamp_point_index(0, 3) == 0
    assert clamp_point_index(7, 3) == 2
    assert clamp_point_index(5, 0) == 0


def test_safe_point_index_clamps_invalid_values() -> None:
    assert safe_point_index(float("nan"), 3, fallback=1) == 1
    assert safe_point_index(None, 3, fallback=2) == 2
    assert safe_point_index(9.8, 3, fallback=0) == 2


def test_snapshot_turn_helpers_distinguish_motion_and_committed_views() -> None:
    assert motion_base_snapshot_turn(5) == 5
    assert committed_snapshot_turn(5) == 6


def test_replay_piece_states_attaches_and_places_piece() -> None:
    plan = TurnPlan()
    plan.piece_id = 0
    plan.cell_id = 4
    target_pose = _pose(0.49, 0.0, 0.025)
    base_piece = _piece(0, 0, 0.40, 0.18, 0.025)

    tcp_positions_by_phase = {
        "home_to_pick": [
            np.array([0.1, 0.1, 0.4]),
            np.array([0.4, 0.18, 0.025]),
        ],
        "pick_to_home": [
            np.array([0.4, 0.18, 0.025]),
            np.array([0.3, 0.1, 0.2]),
        ],
        "home_to_place": [
            np.array([0.3, 0.1, 0.2]),
            np.array([0.49, 0.0, 0.025]),
        ],
        "place_to_home": [
            np.array([0.49, 0.0, 0.025]),
            np.array([0.3, 0.1, 0.2]),
        ],
    }

    pieces = replay_piece_states(
        base_pieces=[base_piece],
        plan=plan,
        target_pose=target_pose,
        attach_distance_threshold=0.07,
        tcp_positions_by_phase=tcp_positions_by_phase,
        selection=ReplaySelection(phase_name="home_to_place", point_index=1),
    )

    piece = pieces[0]
    assert piece.available is False
    assert piece.location == PieceState.LOCATION_BOARD
    assert piece.cell_id == 4
    assert piece.pose.position.x == target_pose.position.x
    assert piece.pose.position.y == target_pose.position.y
    assert piece.pose.position.z == target_pose.position.z


def test_replay_piece_states_shows_attached_piece_mid_turn() -> None:
    plan = TurnPlan()
    plan.piece_id = 0
    plan.cell_id = 4
    target_pose = _pose(0.49, 0.0, 0.025)
    base_piece = _piece(0, 0, 0.40, 0.18, 0.025)

    tcp_positions_by_phase = {name: [] for name in PHASE_NAMES}
    tcp_positions_by_phase["home_to_pick"] = [np.array([0.4, 0.18, 0.025])]
    tcp_positions_by_phase["pick_to_home"] = [np.array([0.33, 0.10, 0.20])]

    pieces = replay_piece_states(
        base_pieces=[base_piece],
        plan=plan,
        target_pose=target_pose,
        attach_distance_threshold=0.07,
        tcp_positions_by_phase=tcp_positions_by_phase,
        selection=ReplaySelection(phase_name="pick_to_home", point_index=0),
    )

    piece = pieces[0]
    assert piece.available is False
    assert piece.location == PieceState.LOCATION_ATTACHED
    assert piece.cell_id == 255
    assert piece.pose.position.x == 0.33
