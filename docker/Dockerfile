# Quadruped navigation sim (MuJoCo) on top of the existing ROS 2 Humble robot image.
FROM syncai-robot-base:latest

USER root
RUN pip3 install --no-cache-dir mujoco==3.15.0 scipy open3d pytest

# Message-only livox_ros_driver2 so /livox/lidar can be published as CustomMsg.
# Optional: if the build fails the sim falls back to PointCloud2 only.
COPY livox_msgs /opt/livox_ws/src/livox_ros_driver2
RUN bash -c "source /opt/ros/humble/setup.bash && cd /opt/livox_ws && \
      (colcon build --cmake-args -DBUILD_TESTING=OFF > /opt/livox_ws/build.log 2>&1 \
       && echo '[ -f /opt/livox_ws/install/setup.bash ] && source /opt/livox_ws/install/setup.bash' >> /etc/bash.bashrc \
       || (tail -20 /opt/livox_ws/build.log; echo 'livox msgs build failed, CustomMsg disabled'))"

# The base image user (uid 1000) matches the host user, so /ws stays writable.
USER syncrobotic
# Same middleware as ../WebRvizLite (Cyclone is already in the base image).
ENV PYTHONUNBUFFERED=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
WORKDIR /ws
