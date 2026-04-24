from __future__ import annotations

import copy
import threading

import numpy as np
import rclpy
import viser
from rclpy.node import Node
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from ttt_interfaces.msg import GameSnapshot, TurnPlan
from ttt_layout import load_layout

from .replay import (
    PHASE_NAMES,
    ReplaySelection,
    VIEW_MODES,
    clone_piece,
    committed_snapshot_turn,
    get_phase_point_count,
    get_phase_trajectory,
    motion_base_snapshot_turn,
    replay_piece_states,
    safe_point_index,
)


class TicTacToeVisualizerNode(Node):
    def __init__(self) -> None:
        super().__init__("ttt_visualizer")
        self.layout = load_layout().workspace
        self.state_lock = threading.Lock()

        self.server = viser.ViserServer(
            host="0.0.0.0", port=8080, label="ISDN3000E Tic-Tac-Toe"
        )
        self.server.scene.add_grid("/grid", width=1.0, height=1.0)

        self.piece_handles: dict[int, object] = {}
        self.snapshot_by_turn: dict[int, GameSnapshot] = {}
        self.plan_by_turn: dict[int, TurnPlan] = {}
        self.latest_snapshot: GameSnapshot | None = None
        self.selected_turn: int | None = None
        self.selected_phase = PHASE_NAMES[0]
        self.selected_view_mode = VIEW_MODES[0]
        self.selected_point_index_by_phase = {phase_name: 0 for phase_name in PHASE_NAMES}
        self.user_has_interacted = False
        self.syncing_gui = False

        urdf = load_robot_description(
            "panda_description",
            load_meshes=True,
            build_scene_graph=True,
            load_collision_meshes=False,
        )
        self.urdf = urdf
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=urdf, load_meshes=True)
        self.actuated_joint_names = list(
            self.viser_urdf.get_actuated_joint_limits().keys()
        )
        self.viser_urdf.update_cfg(
            self._expand_cfg(self.layout.home_joint_state.position)
        )

        for index, pose in enumerate(self.layout.cell_poses):
            self.server.scene.add_box(
                name=f"/board/cell_{index}",
                dimensions=(0.05, 0.05, 0.005),
                position=(pose.position.x, pose.position.y, -0.0025),
                color=(220, 60, 60),
            )

        self.status = self.server.gui.add_markdown("Waiting for match state...")
        with self.server.gui.add_folder("Turn Controls"):
            self.prev_turn_button = self.server.gui.add_button("Previous Turn")
            self.next_turn_button = self.server.gui.add_button("Next Turn")
            self.clear_button = self.server.gui.add_button("Clear")
            self.view_mode_selector = self.server.gui.add_button_group(
                "View Mode", options=VIEW_MODES
            )
            self.phase_selector = self.server.gui.add_button_group(
                "Phase", options=PHASE_NAMES
            )

        with self.server.gui.add_folder("Trajectory Sliders"):
            self.phase_sliders = {
                phase_name: self.server.gui.add_slider(
                    phase_name,
                    min=0.0,
                    max=1.0,
                    step=1.0,
                    initial_value=0.0,
                )
                for phase_name in PHASE_NAMES
            }

        @self.prev_turn_button.on_click
        def _(_event) -> None:
            self._step_turn(-1)

        @self.next_turn_button.on_click
        def _(_event) -> None:
            self._step_turn(1)

        @self.clear_button.on_click
        def _(_event) -> None:
            self._clear_recording()

        @self.view_mode_selector.on_click
        def _(event) -> None:
            if self.syncing_gui:
                return
            self._set_selected_view_mode(str(event.target.value))

        @self.phase_selector.on_click
        def _(event) -> None:
            if self.syncing_gui:
                return
            self._set_selected_phase(str(event.target.value))

        for phase_name, slider in self.phase_sliders.items():

            @slider.on_update
            def _(_event, phase_name: str = phase_name) -> None:
                if self.syncing_gui:
                    return
                self._on_slider_update(phase_name)

        self.snapshot_sub = self.create_subscription(
            GameSnapshot, "/ttt/game_snapshot", self._on_snapshot, 10
        )
        self.plan_sub = self.create_subscription(
            TurnPlan, "/ttt/accepted_turn_plan", self._on_turn_plan, 10
        )
        self._render_pieces(self.layout.initial_pieces)
        self._refresh_gui_and_scene()

    def _on_snapshot(self, msg: GameSnapshot) -> None:
        with self.state_lock:
            snapshot = copy.deepcopy(msg)
            self.latest_snapshot = snapshot
            self.snapshot_by_turn[int(snapshot.turn_index)] = snapshot
        self._refresh_gui_and_scene()

    def _on_turn_plan(self, msg: TurnPlan) -> None:
        with self.state_lock:
            plan = copy.deepcopy(msg)
            self.plan_by_turn[int(plan.turn_index)] = plan
            if self.selected_turn is None:
                self.selected_turn = int(plan.turn_index)
        self._refresh_gui_and_scene()

    def _step_turn(self, direction: int) -> None:
        with self.state_lock:
            self.user_has_interacted = True
            turns = self._available_turns_locked()
            if not turns:
                return
            if self.selected_turn not in turns:
                self.selected_turn = turns[0 if direction >= 0 else -1]
            else:
                current_index = turns.index(self.selected_turn)
                next_index = max(0, min(current_index + direction, len(turns) - 1))
                self.selected_turn = turns[next_index]
        self._refresh_gui_and_scene()

    def _set_selected_phase(self, phase_name: str) -> None:
        with self.state_lock:
            if phase_name not in PHASE_NAMES:
                return
            self.user_has_interacted = True
            self.selected_phase = phase_name
            self.phase_selector.value = phase_name
        self._refresh_gui_and_scene()

    def _set_selected_view_mode(self, view_mode: str) -> None:
        with self.state_lock:
            if view_mode not in VIEW_MODES:
                return
            self.user_has_interacted = True
            self.selected_view_mode = view_mode
            self.view_mode_selector.value = view_mode
        self._refresh_gui_and_scene()

    def _on_slider_update(self, phase_name: str) -> None:
        with self.state_lock:
            self.user_has_interacted = True
            plan = self.plan_by_turn.get(self.selected_turn) if self.selected_turn is not None else None
            point_count = get_phase_point_count(plan, phase_name) if plan is not None else 0
            slider_value = self.phase_sliders[phase_name].value
            clamped_value = safe_point_index(
                slider_value,
                point_count,
                fallback=self.selected_point_index_by_phase[phase_name],
            )
            self.selected_point_index_by_phase[phase_name] = clamped_value
            self.selected_phase = phase_name
            self.selected_view_mode = "motion"
        self._refresh_gui_and_scene()

    def _clear_recording(self) -> None:
        with self.state_lock:
            self.snapshot_by_turn.clear()
            self.plan_by_turn.clear()
            self.latest_snapshot = None
            self.selected_turn = None
            self.selected_phase = PHASE_NAMES[0]
            self.selected_view_mode = VIEW_MODES[0]
            self.selected_point_index_by_phase = {
                phase_name: 0 for phase_name in PHASE_NAMES
            }
            self.user_has_interacted = False
        self._refresh_gui_and_scene()

    def _refresh_gui_and_scene(self) -> None:
        with self.state_lock:
            self._sync_controls_locked()
            status_content, piece_states, robot_cfg = self._build_scene_state_locked()

        self.status.content = status_content
        self.viser_urdf.update_cfg(robot_cfg)
        self._render_pieces(piece_states.values())

    def _sync_controls_locked(self) -> None:
        self.syncing_gui = True
        try:
            turns = self._available_turns_locked()
            if turns and self.selected_turn is None:
                self.selected_turn = turns[0]
            if turns and self.selected_turn not in turns:
                self.selected_turn = turns[-1]

            if (
                not self.user_has_interacted
                and turns
                and self.selected_turn == turns[-1]
                and self.latest_snapshot is not None
                and int(self.latest_snapshot.turn_index)
                >= committed_snapshot_turn(self.selected_turn)
                and self.selected_view_mode == "motion"
            ):
                self.selected_view_mode = "committed_snapshot"

            has_previous = bool(turns and self.selected_turn is not None and turns.index(self.selected_turn) > 0)
            has_next = bool(turns and self.selected_turn is not None and turns.index(self.selected_turn) < len(turns) - 1)
            self.prev_turn_button.disabled = not has_previous
            self.next_turn_button.disabled = not has_next
            self.view_mode_selector.value = self.selected_view_mode
            self.phase_selector.value = self.selected_phase

            plan = self.plan_by_turn.get(self.selected_turn) if self.selected_turn is not None else None
            for phase_name, slider in self.phase_sliders.items():
                if plan is None:
                    slider.disabled = True
                    slider.min = 0.0
                    slider.max = 1.0
                    slider.value = 0.0
                    continue

                point_count = get_phase_point_count(plan, phase_name)
                slider.disabled = point_count == 0
                slider.min = 0.0
                slider.max = float(max(1, point_count - 1))
                clamped_value = safe_point_index(
                    self.selected_point_index_by_phase[phase_name],
                    point_count,
                )
                self.selected_point_index_by_phase[phase_name] = clamped_value
                slider.value = float(clamped_value)
        finally:
            self.syncing_gui = False

    def _build_scene_state_locked(self) -> tuple[str, dict[int, object], np.ndarray]:
        snapshot = self.latest_snapshot
        turns = self._available_turns_locked()
        if not turns or self.selected_turn is None or self.selected_turn not in self.plan_by_turn:
            pieces = self._snapshot_piece_dict(snapshot)
            return (
                self._build_status_markdown(snapshot=snapshot, turns=turns, replay_summary="No accepted turn selected."),
                pieces,
                self._expand_cfg(self.layout.home_joint_state.position),
            )

        plan = self.plan_by_turn[self.selected_turn]
        if self.selected_view_mode == "committed_snapshot":
            committed_snapshot = self.snapshot_by_turn.get(
                committed_snapshot_turn(self.selected_turn)
            )
            pieces = self._snapshot_piece_dict(committed_snapshot or snapshot)
            replay_summary = (
                f"Selected turn: `{self.selected_turn}`  \n"
                f"Selected view: `committed_snapshot`  \n"
                f"Committed snapshot turn: `{committed_snapshot_turn(self.selected_turn)}`  \n"
                f"Piece: `{plan.piece_id}` -> Cell: `{plan.cell_id}`"
            )
            return (
                self._build_status_markdown(
                    snapshot=snapshot,
                    turns=turns,
                    replay_summary=replay_summary,
                ),
                pieces,
                self._expand_cfg(self.layout.home_joint_state.position),
            )

        base_snapshot = self.snapshot_by_turn.get(motion_base_snapshot_turn(self.selected_turn))
        base_pieces = list(base_snapshot.pieces) if base_snapshot is not None else self.layout.initial_pieces
        target_pose = self.layout.cell_poses[int(plan.cell_id)]
        point_index = self.selected_point_index_by_phase[self.selected_phase]

        tcp_positions_by_phase: dict[str, list[np.ndarray]] = {}
        final_cfg = self._expand_cfg(self.layout.home_joint_state.position)
        for phase_name in PHASE_NAMES:
            trajectory = get_phase_trajectory(plan, phase_name)
            tcp_positions: list[np.ndarray] = []
            for point in trajectory.points:
                cfg = self._expand_cfg(point.positions)
                final_cfg = cfg
                tcp_positions.append(self._tcp_position(cfg))
            tcp_positions_by_phase[phase_name] = tcp_positions

        selected_trajectory = get_phase_trajectory(plan, self.selected_phase)
        if selected_trajectory.points:
            clamped_index = safe_point_index(point_index, len(selected_trajectory.points))
            final_cfg = self._expand_cfg(
                selected_trajectory.points[clamped_index].positions
            )
        else:
            clamped_index = 0

        pieces = replay_piece_states(
            base_pieces=base_pieces,
            plan=plan,
            target_pose=target_pose,
            attach_distance_threshold=float(self.layout.attach_distance_threshold),
            tcp_positions_by_phase=tcp_positions_by_phase,
            selection=ReplaySelection(
                phase_name=self.selected_phase,
                point_index=clamped_index,
            ),
        )

        replay_summary = (
            f"Selected turn: `{self.selected_turn}`  \n"
            f"Selected view: `motion`  \n"
            f"Base snapshot turn: `{motion_base_snapshot_turn(self.selected_turn)}`  \n"
            f"Selected phase: `{self.selected_phase}`  \n"
            f"Waypoint: `{clamped_index}` / `{max(0, get_phase_point_count(plan, self.selected_phase) - 1)}`  \n"
            f"Piece: `{plan.piece_id}` -> Cell: `{plan.cell_id}`"
        )
        return (
            self._build_status_markdown(
                snapshot=snapshot,
                turns=turns,
                replay_summary=replay_summary,
            ),
            pieces,
            final_cfg,
        )

    def _build_status_markdown(
        self,
        snapshot: GameSnapshot | None,
        turns: list[int],
        replay_summary: str,
    ) -> str:
        if snapshot is None:
            return (
                "Match state not received yet.  \n"
                f"Recorded turns: `{turns}`  \n"
                f"{replay_summary}"
            )

        current_player = (
            "none"
            if snapshot.current_player == GameSnapshot.NO_PLAYER
            else str(snapshot.current_player)
        )
        return (
            f"Match: `{snapshot.match_id}`  \n"
            f"Latest snapshot turn: `{snapshot.turn_index}`  \n"
            f"Current player: `{current_player}`  \n"
            f"Recorded turns: `{turns}`  \n"
            f"Message: {snapshot.message}  \n\n"
            f"{replay_summary}"
        )

    def _available_turns_locked(self) -> list[int]:
        return sorted(self.plan_by_turn.keys())

    def _snapshot_piece_dict(self, snapshot: GameSnapshot | None) -> dict[int, object]:
        if snapshot is None:
            return {
                piece.piece_id: clone_piece(piece)
                for piece in self.layout.initial_pieces
            }
        return {piece.piece_id: clone_piece(piece) for piece in snapshot.pieces}

    def _render_pieces(self, pieces) -> None:
        for piece in pieces:
            color = (80, 200, 80) if piece.piece_id <= 5 else (80, 140, 255)
            visible = piece.available or piece.location != piece.LOCATION_STOCK
            cube_position = (
                piece.pose.position.x,
                piece.pose.position.y,
                piece.pose.position.z,
            )

            box_handle = self.piece_handles.get(piece.piece_id)
            if box_handle is None:
                box_handle = self.server.scene.add_box(
                    name=f"/pieces/piece_{piece.piece_id}",
                    dimensions=(
                        self.layout.piece_size.x,
                        self.layout.piece_size.y,
                        self.layout.piece_size.z,
                    ),
                    position=cube_position,
                    color=color,
                    visible=visible,
                )
                self.piece_handles[piece.piece_id] = box_handle
                continue

            box_handle.position = cube_position
            box_handle.visible = visible

    def _expand_cfg(self, arm_positions) -> np.ndarray:
        joint_map = dict(zip(self.layout.home_joint_state.name, arm_positions))
        expanded = [
            float(joint_map.get(joint_name, 0.0))
            for joint_name in self.actuated_joint_names
        ]
        return np.array(expanded, dtype=float)

    def _tcp_position(self, cfg: np.ndarray) -> np.ndarray:
        self.urdf.update_cfg(cfg)
        return self.urdf.get_transform("panda_hand_tcp")[:3, 3]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TicTacToeVisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
