from __future__ import annotations

from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import Trigger

from ttt_interfaces.msg import ExecutionResult, GameSnapshot, TurnPlan
from ttt_interfaces.srv import PlanTurn, RegisterPlayer, ReviewTurn
from ttt_layout import load_layout

from .coordinator import MatchCoordinator


class TicTacToeEngineNode(Node):
    def __init__(self) -> None:
        super().__init__("ttt_engine")
        self.declare_parameter("seed", 42)
        self.declare_parameter("autostart", False)
        seed = int(self.get_parameter("seed").value)
        self.autostart = bool(self.get_parameter("autostart").value)

        self.layout_data = load_layout()
        self.coordinator = MatchCoordinator(
            layout=self.layout_data.workspace, seed=seed
        )

        self.snapshot_pub = self.create_publisher(
            GameSnapshot, "/ttt/game_snapshot", 10
        )
        self.accepted_turn_pub = self.create_publisher(
            TurnPlan, "/ttt/accepted_turn_plan", 10
        )
        self.execution_pub = self.create_publisher(
            ExecutionResult, "/ttt/execution_result", 10
        )

        self.register_srv = self.create_service(
            RegisterPlayer, "/ttt/register_player", self._on_register
        )
        self.start_srv = self.create_service(
            Trigger, "/ttt/start_match", self._on_start_match
        )
        self.referee_client = self.create_client(ReviewTurn, "/ttt/review_turn")
        self.player_clients: dict[str, object] = {}
        self.pending_turn_future: Future | None = None
        self.pending_review_future: Future | None = None
        self.timer = self.create_timer(0.2, self._tick)
        self.publish_snapshot()

    def _on_register(
        self, request: RegisterPlayer.Request, response: RegisterPlayer.Response
    ) -> RegisterPlayer.Response:
        success, player_id, message = self.coordinator.register_player(
            request.player_name, request.service_name
        )
        response.success = success
        response.assigned_player_id = player_id
        response.message = message
        if success and self.autostart and self.coordinator.ready_to_start():
            self.coordinator.start_match()
        self.publish_snapshot()
        return response

    def _on_start_match(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if not self.coordinator.ready_to_start():
            response.success = False
            response.message = "Cannot start match until both players are registered."
            return response
        if self.coordinator.status == GameSnapshot.STATUS_PLAYING:
            response.success = False
            response.message = "Match is already running."
            return response
        self.coordinator.start_match()
        self.publish_snapshot()
        response.success = True
        response.message = "Match started."
        return response

    def publish_snapshot(self) -> None:
        self.snapshot_pub.publish(self.coordinator.snapshot_msg())

    def _tick(self) -> None:
        self.publish_snapshot()
        if self.coordinator.status != GameSnapshot.STATUS_PLAYING:
            return
        if (
            self.pending_turn_future is not None
            or self.pending_review_future is not None
        ):
            return

        player = self.coordinator.current_player()
        if player is None:
            return

        client = self.player_clients.get(player.service_name)
        if client is None:
            client = self.create_client(PlanTurn, player.service_name)
            self.player_clients[player.service_name] = client

        if not client.wait_for_service(timeout_sec=0.01):
            return

        request = PlanTurn.Request()
        request.match_id = self.coordinator.match_id
        request.turn_index = self.coordinator.turn_index
        request.player_id = self.coordinator.snapshot_data.current_player
        request.snapshot = self.coordinator.snapshot_msg()
        request.layout = self.layout_data.workspace

        future = client.call_async(request)
        future.add_done_callback(
            partial(self._on_turn_response, player_id=request.player_id)
        )
        self.pending_turn_future = future

    def _on_turn_response(self, future: Future, player_id: int) -> None:
        self.pending_turn_future = None
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.coordinator.mark_failure(f"Player service call failed: {exc}")
            self.publish_snapshot()
            return

        if not response.accepted:
            self.coordinator.mark_failure(
                f"Player_{player_id} failed to provide a plan: {response.message}"
            )
            self.publish_snapshot()
            return

        valid, message = self.coordinator.validate_piece_choice(
            player_id=player_id,
            piece_id=int(response.plan.piece_id),
            cell_id=int(response.plan.cell_id),
        )
        if not valid:
            self.coordinator.mark_failure(message)
            self.publish_snapshot()
            return

        if not self.referee_client.wait_for_service(timeout_sec=0.2):
            self.coordinator.mark_failure("Referee service is unavailable.")
            self.publish_snapshot()
            return

        review_request = ReviewTurn.Request()
        review_request.plan = response.plan
        review_request.snapshot = self.coordinator.snapshot_msg()
        review_request.layout = self.layout_data.workspace
        future = self.referee_client.call_async(review_request)
        future.add_done_callback(
            partial(self._on_review_response, plan=response.plan, player_id=player_id)
        )
        self.pending_review_future = future

    def _on_review_response(
        self, future: Future, plan: TurnPlan, player_id: int
    ) -> None:
        self.pending_review_future = None
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.coordinator.mark_failure(f"Referee service call failed: {exc}")
            self.publish_snapshot()
            return

        self.execution_pub.publish(response.result)
        if not response.result.success:
            self.coordinator.mark_failure(
                f"Player_{player_id} turn rejected: {response.result.message}"
            )
            self.publish_snapshot()
            return

        self.accepted_turn_pub.publish(plan)
        self.coordinator.commit_turn(
            player_id=player_id, piece_id=int(plan.piece_id), cell_id=int(plan.cell_id)
        )
        self.publish_snapshot()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TicTacToeEngineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
