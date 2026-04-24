#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
set -u

sudo rosdep install --from-paths src --ignore-src --rosdistro humble -r -y

# The workspace is bind-mounted and may contain stale colcon artifacts from
# another host/container path. Clean once to avoid symlink-install conflicts.
if [ -d /workspace/build ] || [ -d /workspace/install ] || [ -d /workspace/log ]; then
  rm -rf /workspace/build /workspace/install /workspace/log
fi

colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
