from __future__ import annotations

import os
from pathlib import Path

import pytest
import rclpy

from ttt_bringup.ik_pose_audit import launch_moveit_stack, run_audit, stop_process, build_audit_targets


def test_build_audit_targets_covers_all_cells_and_pieces() -> None:
    targets = build_audit_targets()

    piece_targets = [target for target in targets if target.kind == "piece"]
    cell_targets = [target for target in targets if target.kind == "cell"]

    assert len(piece_targets) == 12
    assert len(cell_targets) == 9
    assert {target.name for target in targets} == {
        *(f"piece_{index}" for index in range(12)),
        *(f"cell_{index}" for index in range(9)),
    }


@pytest.mark.integration
def test_all_cell_and_piece_targets_have_safe_ik_solutions() -> None:
    workspace_root = Path(os.environ.get("COLCON_CURRENT_PREFIX", "/workspace/install")).parent
    process = launch_moveit_stack(workspace_root)
    rclpy.init()
    try:
        results = run_audit()
    finally:
        rclpy.shutdown()
        stop_process(process)

    assert len(results) == 21
