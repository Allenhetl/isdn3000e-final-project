# ISDN3000E Final Project

ROS 2 Humble Tic-Tac-Toe final project starter for `isdn3000e`.

## Before You Start

- Fork this repository to your own GitHub account.
- Keep the course repository as your `upstream` remote.
- Regularly fetch and merge/rebase from `upstream` so you stay on the latest starter code, fixes, and clarifications released during the course.

Typical setup:

```bash
git remote add upstream https://github.com/RoboFab/isdn3000e-final-project.git
git fetch upstream
```

Typical update flow later:

```bash
git fetch upstream
git merge upstream/public-student
```

We might update or fix the bug of this repo. Please instruct your coding agent always keep your upstream as the latest.


## TL;DR

- You will implement the student player in:
  - `src/ttt_player/src/student_player_node.cpp`
- The system already provides:
  - game engine
  - referee
  - workspace layout
  - browser visualizer
  - a scripted dummy opponent
- Your player must return a valid `TurnPlan` for each turn.

## What You Need To Do

Implement the missing logic in:

- `compute_ik(...)`
- `handle_plan_turn(...)`
- `find_piece_pose(...)`

inside:

- `src/ttt_player/src/student_player_node.cpp`

Read the TODO comments in that file carefully.

## Key Files

- `src/ttt_player/src/student_player_node.cpp`
  - your main implementation target
- `src/ttt_interfaces/msg/TurnPlan.msg`
  - required output plan format
- `src/ttt_interfaces/msg/GameSnapshot.msg`
  - current board state and legal actions
- `src/ttt_interfaces/msg/WorkspaceLayout.msg`
  - board cell poses and initial piece poses
- `src/ttt_layout/config/board_layout.yaml`
  - workspace geometry
- `src/ttt_bringup/launch/verify_dummy_first.launch.py`
  - launch file for a local check

## Build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run

Launch the stack:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ttt_bringup verify_dummy_first.launch.py
```

Start the match in another terminal:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 service call /ttt/start_match std_srvs/srv/Trigger
```

Open the visualizer:

- `http://localhost:8080`

## Minimal Project Flow

1. Your player registers with the engine.
2. The engine sends a `PlanTurn` request.
3. Your player chooses a legal piece and cell.
4. Your player computes pick/place joint goals.
5. Your player returns the 4 required trajectories.
6. The referee validates the plan.
7. The engine commits the turn if accepted.

## Hints

- Use `request->snapshot.legal_actions` to find legal cells.
- Use `request->snapshot.pieces` to inspect available pieces.
- Use `request->layout.cell_poses` for target cell poses.
- Use `request->layout.initial_pieces` and/or current snapshot data to find stock piece poses.
- Build `home_to_pick`, `pick_to_home`, `home_to_place`, and `place_to_home` carefully.
- Keep your plan consistent with the Panda joint naming expected by the referee.

## Submission Reminder

- Do not modify interface definitions unless your instructor explicitly asks you to.
- Keep your implementation inside the student player package unless instructed otherwise.
