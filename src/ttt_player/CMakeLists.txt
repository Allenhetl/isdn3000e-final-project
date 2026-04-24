cmake_minimum_required(VERSION 3.8)
project(ttt_player)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(ttt_interfaces REQUIRED)

add_executable(student_player_node src/student_player_node.cpp)
ament_target_dependencies(student_player_node rclcpp geometry_msgs moveit_msgs sensor_msgs trajectory_msgs ttt_interfaces)

install(TARGETS
  student_player_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
