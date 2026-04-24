from __future__ import annotations

from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

import rclpy

from ttt_interfaces.msg import ExecutionResult, GameSnapshot, TurnPlan
from ttt_interfaces.srv import ReviewTurn
from ttt_layout import load_layout
from ttt_referee.referee_node import TicTacToeRefereeNode


EXPECTED_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
HOME = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]
PICK_PIECE_0 = [0.2131, 0.4038, 0.1566, -2.2588, 0.0466, 2.1227, 0.7854]
PLACE_CELL_5 = [0.0, 0.5498, 0.0, -1.9790, -0.0002, 2.2333, 0.7854]
WRONG_PICK_AT_CELL_5 = PLACE_CELL_5


def _make_trajectory(start: list[float], goal: list[float]) -> RobotTrajectory:
    point_start = JointTrajectoryPoint()
    point_start.positions = list(start)
    point_goal = JointTrajectoryPoint()
    point_goal.positions = list(goal)

    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = list(EXPECTED_JOINTS)
    trajectory.joint_trajectory.points = [point_start, point_goal]
    return trajectory


def _build_snapshot() -> GameSnapshot:
    layout = load_layout().workspace
    snapshot = GameSnapshot()
    snapshot.current_player = 0
    snapshot.winner = GameSnapshot.NO_PLAYER
    snapshot.legal_actions = [1] * 9
    snapshot.pieces = list(layout.initial_pieces)
    return snapshot


def _build_request(*, pick_goal: list[float], place_goal: list[float]) -> ReviewTurn.Request:
    layout = load_layout().workspace
    request = ReviewTurn.Request()
    request.layout = layout
    request.snapshot = _build_snapshot()

    plan = TurnPlan()
    plan.player_id = 0
    plan.piece_id = 0
    plan.cell_id = 5
    plan.home_to_pick = _make_trajectory(HOME, pick_goal)
    plan.pick_to_home = _make_trajectory(pick_goal, HOME)
    plan.home_to_place = _make_trajectory(HOME, place_goal)
    plan.place_to_home = _make_trajectory(place_goal, HOME)
    request.plan = plan
    return request


def test_referee_accepts_plan_with_endpoints_near_piece_and_cell() -> None:
    rclpy.init()
    node = TicTacToeRefereeNode()
    try:
        request = _build_request(pick_goal=PICK_PIECE_0, place_goal=PLACE_CELL_5)
        response = node._on_review(request, ReviewTurn.Response())

        assert response.result.success is True
        assert response.result.failure_reason == ExecutionResult.REASON_NONE
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_referee_rejects_plan_when_pick_endpoint_misses_declared_piece() -> None:
    rclpy.init()
    node = TicTacToeRefereeNode()
    try:
        request = _build_request(
            pick_goal=WRONG_PICK_AT_CELL_5,
            place_goal=PLACE_CELL_5,
        )
        response = node._on_review(request, ReviewTurn.Response())

        assert response.result.success is False
        assert (
            response.result.failure_reason
            == ExecutionResult.REASON_MALFORMED_TRAJECTORY
        )
        assert "home_to_pick" in response.result.message
    finally:
        node.destroy_node()
        rclpy.shutdown()
