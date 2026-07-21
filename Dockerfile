# ROS Noetic is EOL. Pin the final official multi-architecture Focal image index
# so a future tag change cannot silently alter the control environment.
FROM ros:noetic-ros-base-focal@sha256:72b8bc59035dc0a5b8e07aae28c16caa84192971d72d207c72ed734fb1d5e97d

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-evdev \
    python3-numpy \
    python3-pip \
    python3-setuptools \
    python3-sympy \
    ros-noetic-geometry-msgs \
    ros-noetic-rosgraph \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-noetic.txt /tmp/requirements-noetic.txt
RUN pip3 install --no-cache-dir --disable-pip-version-check \
    --require-hashes \
    -r /tmp/requirements-noetic.txt

COPY . /catkin_ws/src/giraf_optitrak_teleop
WORKDIR /catkin_ws
RUN source /opt/ros/noetic/setup.bash \
    && catkin_make -DCATKIN_ENABLE_TESTING=OFF \
    && source /catkin_ws/devel/setup.bash \
    && python3 -m unittest discover \
        -s /catkin_ws/src/giraf_optitrak_teleop/tests \
        -p 'test_teleop_core.py' \
        -v

ENTRYPOINT ["/bin/bash", "-c", "source /catkin_ws/devel/setup.bash && exec \"$@\"", "--"]
CMD ["roslaunch", "giraf_optitrak_teleop", "giraf_optitrak_teleop.launch"]
