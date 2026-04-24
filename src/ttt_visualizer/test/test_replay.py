from __future__ import annotations

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


def _pose_with_orientation(
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> Pose:
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
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


def test_replay_piece_states_places_piece_in_place_to_home() -> None:
    plan = TurnPlan()
    plan.piece_id = 0
    plan.cell_id = 4
    target_pose = _pose(0.49, 0.0, 0.025)
    base_piece = _piece(0, 0, 0.40, 0.18, 0.025)

    tcp_poses_by_phase = {
        "home_to_pick": [
            _pose(0.1, 0.1, 0.4),
            _pose(0.4, 0.18, 0.025),
        ],
        "pick_to_home": [
            _pose(0.4, 0.18, 0.025),
            _pose(0.3, 0.1, 0.2),
        ],
        "home_to_place": [
            _pose(0.3, 0.1, 0.2),
            _pose(0.49, 0.0, 0.025),
        ],
        "place_to_home": [
            _pose(0.49, 0.0, 0.025),
            _pose(0.3, 0.1, 0.2),
        ],
    }

    pieces = replay_piece_states(
        base_pieces=[base_piece],
        plan=plan,
        target_pose=target_pose,
        tcp_poses_by_phase=tcp_poses_by_phase,
        selection=ReplaySelection(phase_name="place_to_home", point_index=1),
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

    tcp_poses_by_phase = {name: [] for name in PHASE_NAMES}
    tcp_poses_by_phase["home_to_pick"] = [_pose(0.4, 0.18, 0.025)]
    tcp_poses_by_phase["pick_to_home"] = [
        _pose_with_orientation(0.33, 0.10, 0.20, 0.2, 0.3, 0.4, 0.84)
    ]

    pieces = replay_piece_states(
        base_pieces=[base_piece],
        plan=plan,
        target_pose=target_pose,
        tcp_poses_by_phase=tcp_poses_by_phase,
        selection=ReplaySelection(phase_name="pick_to_home", point_index=0),
    )

    piece = pieces[0]
    assert piece.available is False
    assert piece.location == PieceState.LOCATION_ATTACHED
    assert piece.cell_id == 255
    assert piece.pose.position.x == 0.33
    assert piece.pose.orientation.x == 0.2
    assert piece.pose.orientation.y == 0.3
    assert piece.pose.orientation.z == 0.4
    assert piece.pose.orientation.w == 0.84


def test_replay_piece_states_keeps_piece_in_base_state_in_home_to_pick() -> None:
    plan = TurnPlan()
    plan.piece_id = 0
    plan.cell_id = 4
    target_pose = _pose(0.49, 0.0, 0.025)
    base_piece = _piece(0, 0, 0.40, 0.18, 0.025)

    tcp_poses_by_phase = {name: [] for name in PHASE_NAMES}
    tcp_poses_by_phase["home_to_pick"] = [
        _pose(0.4, 0.18, 0.025),
        _pose(0.41, 0.17, 0.03),
    ]

    pieces = replay_piece_states(
        base_pieces=[base_piece],
        plan=plan,
        target_pose=target_pose,
        tcp_poses_by_phase=tcp_poses_by_phase,
        selection=ReplaySelection(phase_name="home_to_pick", point_index=1),
    )

    piece = pieces[0]
    assert piece.available is True
    assert piece.location == PieceState.LOCATION_STOCK
    assert piece.cell_id == 255
    assert piece.pose.position.x == 0.40
    assert piece.pose.position.y == 0.18
    assert piece.pose.position.z == 0.025


def test_replay_piece_states_shows_attached_piece_in_home_to_place() -> None:
    plan = TurnPlan()
    plan.piece_id = 0
    plan.cell_id = 4
    target_pose = _pose(0.49, 0.0, 0.025)
    base_piece = _piece(0, 0, 0.40, 0.18, 0.025)

    tcp_poses_by_phase = {name: [] for name in PHASE_NAMES}
    tcp_poses_by_phase["home_to_place"] = [
        _pose_with_orientation(0.31, 0.09, 0.21, 0.0, 0.0, 0.0, 1.0),
        _pose_with_orientation(0.35, 0.06, 0.16, -0.1, 0.4, 0.2, 0.88),
    ]

    pieces = replay_piece_states(
        base_pieces=[base_piece],
        plan=plan,
        target_pose=target_pose,
        tcp_poses_by_phase=tcp_poses_by_phase,
        selection=ReplaySelection(phase_name="home_to_place", point_index=1),
    )

    piece = pieces[0]
    assert piece.available is False
    assert piece.location == PieceState.LOCATION_ATTACHED
    assert piece.cell_id == 255
    assert piece.pose.position.x == 0.35
    assert piece.pose.position.y == 0.06
    assert piece.pose.position.z == 0.16
    assert piece.pose.orientation.x == -0.1
    assert piece.pose.orientation.y == 0.4
    assert piece.pose.orientation.z == 0.2
    assert piece.pose.orientation.w == 0.88
