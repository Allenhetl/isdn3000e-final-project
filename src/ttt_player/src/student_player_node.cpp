#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "ttt_interfaces/msg/game_snapshot.hpp"
#include "ttt_interfaces/msg/turn_plan.hpp"
#include "ttt_interfaces/msg/workspace_layout.hpp"
#include "ttt_interfaces/srv/plan_turn.hpp"
#include "ttt_interfaces/srv/register_player.hpp"

using namespace std::chrono_literals;

namespace {

std::vector<std::string> panda_joint_names() {
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
          "panda_joint5", "panda_joint6", "panda_joint7"};
}

const std::vector<double> kHomePositions = {0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398};

const std::vector<std::pair<uint8_t, uint8_t>> kScriptedMoves = {
    {6, 2},
    {7, 4},
    {8, 6},
};

// link8 orientation quaternion (x, y, z, w) for gripper pointing down
constexpr double kLink8QuatX = 0.9238795325112867;
constexpr double kLink8QuatY = -0.3826834323650898;
constexpr double kLink8QuatZ = 0.0;
constexpr double kLink8QuatW = 0.0;
// Fixed Z offset from panda_link8 to panda_hand_tcp
constexpr double kLink8TcpZOffset = 0.1034;

geometry_msgs::msg::Pose link8_pose_from_tcp_target(double x, double y, double z) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z + kLink8TcpZOffset;
  pose.orientation.x = kLink8QuatX;
  pose.orientation.y = kLink8QuatY;
  pose.orientation.z = kLink8QuatZ;
  pose.orientation.w = kLink8QuatW;
  return pose;
}

trajectory_msgs::msg::JointTrajectoryPoint make_point(
    const std::vector<double> &positions,
    double time_sec) {
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = positions;
  const auto whole_seconds = static_cast<int32_t>(std::floor(time_sec));
  point.time_from_start.sec = whole_seconds;
  point.time_from_start.nanosec =
      static_cast<uint32_t>((time_sec - static_cast<double>(whole_seconds)) * 1e9);
  return point;
}

moveit_msgs::msg::RobotTrajectory make_three_point_trajectory(
    const std::vector<double> &start_positions,
    const std::vector<double> &end_positions,
    double end_time_sec) {
  moveit_msgs::msg::RobotTrajectory trajectory;
  trajectory.joint_trajectory.joint_names = panda_joint_names();

  const std::vector<double> midpoint = [&]() {
    std::vector<double> result;
    result.reserve(start_positions.size());
    for (size_t index = 0; index < start_positions.size(); ++index) {
      result.push_back((start_positions[index] + end_positions[index]) * 0.5);
    }
    return result;
  }();

  trajectory.joint_trajectory.points.push_back(make_point(start_positions, 0.0));
  trajectory.joint_trajectory.points.push_back(make_point(midpoint, end_time_sec * 0.5));
  trajectory.joint_trajectory.points.push_back(make_point(end_positions, end_time_sec));
  return trajectory;
}

// ------------------------------------------------------------------
// Game logic (Minimax with alpha-beta pruning)
// ------------------------------------------------------------------

constexpr int kWinScore = 10000;
constexpr int kLossScore = -10000;
constexpr int kAlphaInit = -100000;
constexpr int kBetaInit = 100000;
constexpr uint8_t kSentinelNoCell = 0xFF;

// Indexed by cell_id; lower value = higher tie-break preference.
// Derived from preference order [4, 0, 2, 6, 8, 1, 3, 5, 7]
// (center > corners > edges).
constexpr std::array<int, 9> kCellTieBreakRank = {1, 5, 2, 6, 0, 7, 3, 8, 4};

// 8 win lines: 3 rows, 3 cols, 2 diagonals. Order copied verbatim from
// ttt_game/pettingzoo_adapter.py::_check_winner.
constexpr std::array<std::array<uint8_t, 3>, 8> kWinLines = {{
    {{0, 1, 2}}, {{3, 4, 5}}, {{6, 7, 8}},
    {{0, 3, 6}}, {{1, 4, 7}}, {{2, 5, 8}},
    {{0, 4, 8}}, {{2, 4, 6}},
}};

// Returns 1 or 2 if that player has won; 0 otherwise (no winner / not terminal).
uint8_t check_winner(const std::array<uint8_t, 9> &board) {
  for (const auto &line : kWinLines) {
    const uint8_t a = board[line[0]];
    if (a == 0) continue;
    if (a == board[line[1]] && a == board[line[2]]) {
      return a;
    }
  }
  return 0;
}

bool board_is_full(const std::array<uint8_t, 9> &board) {
  for (uint8_t v : board) {
    if (v == 0) return false;
  }
  return true;
}

// Score from `my_player_id`'s perspective.
int minimax(std::array<uint8_t, 9> &board,
            uint8_t to_move,
            uint8_t my_player_id,
            int depth,
            int alpha,
            int beta) {
  const uint8_t winner = check_winner(board);
  if (winner == my_player_id) return kWinScore - depth;
  if (winner != 0) return kLossScore + depth;
  if (board_is_full(board)) return 0;

  const uint8_t next_to_move = static_cast<uint8_t>(3 - to_move);
  const bool maximizing = (to_move == my_player_id);

  if (maximizing) {
    int best = kAlphaInit;
    for (int cell = 0; cell < 9; ++cell) {
      if (board[cell] != 0) continue;
      board[cell] = to_move;
      const int value = minimax(board, next_to_move, my_player_id, depth + 1, alpha, beta);
      board[cell] = 0;
      if (value > best) best = value;
      if (best > alpha) alpha = best;
      if (beta <= alpha) break;
    }
    return best;
  } else {
    int best = kBetaInit;
    for (int cell = 0; cell < 9; ++cell) {
      if (board[cell] != 0) continue;
      board[cell] = to_move;
      const int value = minimax(board, next_to_move, my_player_id, depth + 1, alpha, beta);
      board[cell] = 0;
      if (value < best) best = value;
      if (best < beta) beta = best;
      if (beta <= alpha) break;
    }
    return best;
  }
}

// Root-level argmax with deterministic tie-break (kCellTieBreakRank).
// Returns 0..8, or kSentinelNoCell if no legal move available.
uint8_t minimax_best_move(const ttt_interfaces::msg::GameSnapshot &snapshot,
                          uint8_t my_player_id) {
  std::array<uint8_t, 9> board{};
  for (size_t i = 0; i < 9; ++i) {
    board[i] = snapshot.board[i];
  }

  int best_score = kAlphaInit - 1;
  int best_rank = std::numeric_limits<int>::max();
  uint8_t best_cell = kSentinelNoCell;

  const uint8_t opponent = static_cast<uint8_t>(3 - my_player_id);

  for (int cell = 0; cell < 9; ++cell) {
    if (snapshot.legal_actions[cell] != 1) continue;
    if (board[cell] != 0) continue;  // defensive: should never happen if legal

    board[cell] = my_player_id;
    const int value = minimax(board, opponent, my_player_id,
                              /*depth=*/1, kAlphaInit, kBetaInit);
    board[cell] = 0;

    const int rank = kCellTieBreakRank[cell];
    if (value > best_score || (value == best_score && rank < best_rank)) {
      best_score = value;
      best_rank = rank;
      best_cell = static_cast<uint8_t>(cell);
    }
  }

  return best_cell;
}

// Sort own available pieces by ascending distance to the Panda base
// (sqrt(x^2 + y^2)). No hardcoded piece_id list — works for both
// player_0 and player_1 because their geometries are mirrored.
std::vector<uint8_t> rank_pieces_by_distance(
    const ttt_interfaces::msg::GameSnapshot &snapshot,
    uint8_t my_player_id) {
  struct Entry {
    uint8_t piece_id;
    double distance;
  };
  std::vector<Entry> entries;
  entries.reserve(snapshot.pieces.size());
  for (const auto &piece : snapshot.pieces) {
    if (!piece.available) continue;
    if (piece.owner != my_player_id) continue;
    const double dx = piece.pose.position.x;
    const double dy = piece.pose.position.y;
    entries.push_back({piece.piece_id, std::sqrt(dx * dx + dy * dy)});
  }
  std::sort(entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) { return a.distance < b.distance; });
  std::vector<uint8_t> result;
  result.reserve(entries.size());
  for (const auto &e : entries) result.push_back(e.piece_id);
  return result;
}

}  // namespace

class StudentPlayerNode : public rclcpp::Node {
 public:
  StudentPlayerNode() : Node("student_player") {
    this->declare_parameter<std::string>("player_name", this->get_name());
    this->declare_parameter<std::string>("plan_turn_service", "/student_player/plan_turn");

    player_name_ = this->get_parameter("player_name").as_string();
    plan_turn_service_ = this->get_parameter("plan_turn_service").as_string();

    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    register_client_ =
        this->create_client<ttt_interfaces::srv::RegisterPlayer>("/ttt/register_player");

    ik_client_ = this->create_client<moveit_msgs::srv::GetPositionIK>(
        "/compute_ik",
        rmw_qos_profile_services_default,
        cb_group_);

    plan_turn_service_server_ = this->create_service<ttt_interfaces::srv::PlanTurn>(
        plan_turn_service_,
        std::bind(&StudentPlayerNode::handle_plan_turn, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default,
        cb_group_);

    register_timer_ =
        this->create_wall_timer(500ms, std::bind(&StudentPlayerNode::try_register, this));
  }

 private:
  void try_register() {
    if (registered_ || registration_in_flight_) {
      return;
    }
    if (!register_client_->wait_for_service(100ms)) {
      return;
    }

    auto request = std::make_shared<ttt_interfaces::srv::RegisterPlayer::Request>();
    request->player_name = player_name_;
    request->service_name = plan_turn_service_;
    registration_in_flight_ = true;

    register_client_->async_send_request(
        request,
        [this](rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedFuture future) {
          registration_in_flight_ = false;
          try {
            const auto response = future.get();
            if (!response->success) {
              RCLCPP_WARN(this->get_logger(), "Registration rejected: %s",
                          response->message.c_str());
              return;
            }
            registered_ = true;
            player_id_ = response->assigned_player_id;
            RCLCPP_INFO(this->get_logger(), "Registered as player_%u.", player_id_);
            register_timer_->cancel();
          } catch (const std::exception &exc) {
            RCLCPP_ERROR(this->get_logger(), "Registration failed: %s", exc.what());
          }
        });
  }

  // ----------------------------------------------------------------
  // IK helpers
  // ----------------------------------------------------------------

  std::optional<std::vector<double>> compute_ik(
      const geometry_msgs::msg::Pose &target_pose,
      const std::vector<double> &seed_positions) {
    auto request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
    request->ik_request.group_name = "panda_arm";
    request->ik_request.pose_stamped.header.frame_id = "panda_link0";
    request->ik_request.pose_stamped.pose = target_pose;
    request->ik_request.timeout.sec = 5;
    request->ik_request.avoid_collisions = false;

    moveit_msgs::msg::RobotState seed_state;
    seed_state.joint_state.name = panda_joint_names();
    seed_state.joint_state.position = seed_positions;
    request->ik_request.robot_state = seed_state;

    // Synchronous wait inside a service callback: must NOT use
    // spin_until_future_complete (would re-enter the executor and deadlock,
    // even with Reentrant callback group). Poll wait_for(10ms) until a 6s
    // deadline elapses (1s margin over ik_request.timeout = 5s).
    auto future = ik_client_->async_send_request(request);
    const auto deadline = std::chrono::steady_clock::now() + 6s;
    while (rclcpp::ok()) {
      if (future.wait_for(10ms) == std::future_status::ready) break;
      if (std::chrono::steady_clock::now() >= deadline) {
        return std::nullopt;
      }
    }
    if (!rclcpp::ok()) return std::nullopt;

    auto result = future.get();
    if (!result) return std::nullopt;
    // MoveItErrorCodes::SUCCESS == 1
    if (result->error_code.val != 1) return std::nullopt;

    // The returned joint_state often includes the two finger joints and
    // the 7 arm joints in an unspecified order. Index by name and pull
    // out panda_joint1..7 strictly in that order.
    const auto &sol = result->solution.joint_state;
    if (sol.name.size() != sol.position.size()) return std::nullopt;
    std::unordered_map<std::string, double> by_name;
    by_name.reserve(sol.name.size());
    for (size_t i = 0; i < sol.name.size(); ++i) {
      by_name.emplace(sol.name[i], sol.position[i]);
    }
    std::vector<double> arm_joints;
    arm_joints.reserve(7);
    for (const auto &name : panda_joint_names()) {
      auto it = by_name.find(name);
      if (it == by_name.end()) return std::nullopt;
      arm_joints.push_back(it->second);
    }
    if (arm_joints.size() != 7) return std::nullopt;
    return arm_joints;
  }

  // Multi-seed retry: home -> last_solution_ -> 3x home + N(0, 0.1 rad).
  // Updates last_solution_ on first success. Returns nullopt if all 5
  // seeds fail. No logging here — caller decides whether failure of an
  // entire piece warrants a WARN.
  std::optional<std::vector<double>> compute_ik_robust(
      const geometry_msgs::msg::Pose &target_pose) {
    std::vector<std::vector<double>> seeds;
    seeds.reserve(5);
    seeds.push_back(kHomePositions);
    seeds.push_back(last_solution_);

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<double> noise(0.0, 0.1);
    for (int attempt = 0; attempt < 3; ++attempt) {
      std::vector<double> perturbed = kHomePositions;
      for (auto &v : perturbed) v += noise(rng);
      seeds.push_back(std::move(perturbed));
    }

    for (const auto &seed : seeds) {
      auto solution = compute_ik(target_pose, seed);
      if (solution) {
        last_solution_ = *solution;
        return solution;
      }
    }
    return std::nullopt;
  }

  // ----------------------------------------------------------------
  // Turn planning
  // ----------------------------------------------------------------

  void handle_plan_turn(
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> request,
      std::shared_ptr<ttt_interfaces::srv::PlanTurn::Response> response) {
    try {
      handle_plan_turn_impl(request, response);
    } catch (const std::exception &e) {
      response->accepted = false;
      response->message = std::string("exception in handle_plan_turn: ") + e.what();
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      dump_failure_context(request);
    } catch (...) {
      response->accepted = false;
      response->message = "unknown non-std exception in handle_plan_turn";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      dump_failure_context(request);
    }
  }

  void handle_plan_turn_impl(
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> request,
      std::shared_ptr<ttt_interfaces::srv::PlanTurn::Response> response) {
    if (request->player_id != player_id_) {
      response->accepted = false;
      response->message = "player_id mismatch: req=" +
                          std::to_string(static_cast<unsigned>(request->player_id)) +
                          " self=" + std::to_string(static_cast<unsigned>(player_id_));
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      dump_failure_context(request);
      return;
    }

    if (!ik_client_->wait_for_service(2s)) {
      response->accepted = false;
      response->message = "compute_ik service not available after 2s";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      dump_failure_context(request);
      return;
    }

    const uint8_t cell_id = minimax_best_move(request->snapshot, player_id_);
    if (cell_id == kSentinelNoCell) {
      response->accepted = false;
      response->message = "no legal cell available";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      dump_failure_context(request);
      return;
    }
    RCLCPP_INFO(this->get_logger(), "turn=%u: minimax chose cell=%u",
                static_cast<unsigned>(request->turn_index),
                static_cast<unsigned>(cell_id));

    if (static_cast<size_t>(cell_id) >= request->layout.cell_poses.size()) {
      response->accepted = false;
      response->message = "cell_id out of range for layout.cell_poses";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      dump_failure_context(request);
      return;
    }

    const auto ranked = rank_pieces_by_distance(request->snapshot, player_id_);
    if (ranked.empty()) {
      response->accepted = false;
      response->message = "no available piece for player_" +
                          std::to_string(static_cast<unsigned>(player_id_));
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      dump_failure_context(request);
      return;
    }
    {
      std::ostringstream oss;
      oss << "ranked pieces for cell " << static_cast<unsigned>(cell_id) << ": [";
      for (size_t i = 0; i < ranked.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << static_cast<unsigned>(ranked[i]);
      }
      oss << "]";
      RCLCPP_INFO(this->get_logger(), "%s", oss.str().c_str());
    }

    const auto &cell_pose = request->layout.cell_poses[cell_id];
    const auto place_link8 = link8_pose_from_tcp_target(
        cell_pose.position.x, cell_pose.position.y, cell_pose.position.z);

    uint8_t chosen_piece = 0;
    std::vector<double> pick_joints;
    std::vector<double> place_joints;
    bool found = false;

    for (uint8_t piece_id : ranked) {
      const auto piece_pose_opt = find_piece_pose(request->snapshot, piece_id);
      if (!piece_pose_opt) {
        RCLCPP_WARN(this->get_logger(),
                    "piece %u not found in snapshot, trying next",
                    static_cast<unsigned>(piece_id));
        continue;
      }
      const auto pick_link8 = link8_pose_from_tcp_target(
          piece_pose_opt->position.x,
          piece_pose_opt->position.y,
          piece_pose_opt->position.z);

      auto pick_sol = compute_ik_robust(pick_link8);
      const bool pick_ok = pick_sol.has_value();
      auto place_sol = pick_ok ? compute_ik_robust(place_link8)
                               : std::optional<std::vector<double>>{};
      const bool place_ok = place_sol.has_value();

      if (pick_ok && place_ok) {
        chosen_piece = piece_id;
        pick_joints = std::move(*pick_sol);
        place_joints = std::move(*place_sol);
        found = true;
        RCLCPP_INFO(this->get_logger(),
                    "piece %u IK ok (pick norm=%.3f, place norm=%.3f)",
                    static_cast<unsigned>(piece_id),
                    vector_l2_norm(pick_joints),
                    vector_l2_norm(place_joints));
        break;
      }
      RCLCPP_WARN(this->get_logger(),
                  "piece %u IK failed (pick=%s/place=%s), trying next",
                  static_cast<unsigned>(piece_id),
                  pick_ok ? "ok" : "fail",
                  place_ok ? "ok" : "fail");
    }

    if (!found) {
      response->accepted = false;
      response->message = "IK failed for cell " +
                          std::to_string(static_cast<unsigned>(cell_id)) +
                          " after " + std::to_string(ranked.size()) + " pieces tried";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      dump_failure_context(request);
      return;
    }

    ttt_interfaces::msg::TurnPlan plan;
    plan.match_id = request->match_id;
    plan.turn_index = request->turn_index;
    plan.player_id = request->player_id;
    plan.piece_id = chosen_piece;
    plan.cell_id = cell_id;
    plan.home_to_pick = make_three_point_trajectory(kHomePositions, pick_joints, 1.5);
    plan.pick_to_home = make_three_point_trajectory(pick_joints, kHomePositions, 1.5);
    plan.home_to_place = make_three_point_trajectory(kHomePositions, place_joints, 1.5);
    plan.place_to_home = make_three_point_trajectory(place_joints, kHomePositions, 1.5);

    response->plan = plan;
    response->accepted = true;
    response->message = "ok";
    RCLCPP_INFO(this->get_logger(),
                "final plan: piece=%u -> cell=%u, 4 trajectories built",
                static_cast<unsigned>(chosen_piece),
                static_cast<unsigned>(cell_id));
  }

  static double vector_l2_norm(const std::vector<double> &v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
  }

  void dump_failure_context(
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> &request) {
    try {
      std::ostringstream oss;
      oss << "[failure dump] turn=" << request->turn_index
          << " player_id=" << static_cast<unsigned>(request->player_id)
          << " self=" << static_cast<unsigned>(player_id_)
          << " board=[";
      for (size_t i = 0; i < request->snapshot.board.size(); ++i) {
        if (i > 0) oss << ",";
        oss << static_cast<unsigned>(request->snapshot.board[i]);
      }
      oss << "] legal=[";
      for (size_t i = 0; i < request->snapshot.legal_actions.size(); ++i) {
        if (i > 0) oss << ",";
        oss << static_cast<unsigned>(request->snapshot.legal_actions[i]);
      }
      oss << "] available_pieces=[";
      bool first = true;
      for (const auto &p : request->snapshot.pieces) {
        if (!p.available) continue;
        if (p.owner != player_id_) continue;
        if (!first) oss << ",";
        first = false;
        oss << static_cast<unsigned>(p.piece_id);
      }
      oss << "]";
      RCLCPP_ERROR(this->get_logger(), "%s", oss.str().c_str());
    } catch (...) {
      // dump itself must never throw — caller is already on a failure path.
      RCLCPP_ERROR(this->get_logger(), "[failure dump] (formatting failed)");
    }
  }

  static std::optional<geometry_msgs::msg::Pose> find_piece_pose(
      const ttt_interfaces::msg::GameSnapshot &snapshot,
      uint8_t piece_id) {
    for (const auto &piece : snapshot.pieces) {
      if (piece.piece_id == piece_id) {
        return piece.pose;
      }
    }
    return std::nullopt;
  }

  std::string player_name_;
  std::string plan_turn_service_;
  bool registered_{false};
  bool registration_in_flight_{false};
  uint8_t player_id_{255};
  // Cached last successful IK solution; seeds round-2 of compute_ik_robust.
  // Initialized to home so the first turn naturally falls back to seed-1
  // (home) without reading uninitialized values. Engine schedules
  // plan_turn serially (engine_node.py pending_turn_future), so no mutex
  // is needed here.
  std::vector<double> last_solution_ = kHomePositions;

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedPtr register_client_;
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
  rclcpp::Service<ttt_interfaces::srv::PlanTurn>::SharedPtr plan_turn_service_server_;
  rclcpp::TimerBase::SharedPtr register_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StudentPlayerNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
