FROM ros:noetic-ros-base-focal

ARG PARENTDIR

WORKDIR /root/catkin_ws

USER root

RUN apt update && apt install -y ranger git vim curl &&\
    #sh -c 'echo "deb http://packages.ros.org/ros/ubuntu \
    #$(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
    #curl -s \
    #https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | \
    #sudo apt-key add - \ &&
    #apt update &&\
    DEBIAN_FRONTEND=noninteractive apt install -y ros-noetic-joy ros-noetic-teleop-twist-joy \
    ros-noetic-teleop-twist-keyboard ros-noetic-laser-proc \
    ros-noetic-rgbd-launch ros-noetic-rosserial-arduino \
    ros-noetic-rosserial-python ros-noetic-rosserial-client \
    ros-noetic-rosserial-msgs ros-noetic-amcl ros-noetic-map-server \
    ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro \
    ros-noetic-compressed-image-transport ros-noetic-rqt* \
    ros-noetic-rviz ros-noetic-gmapping \
    ros-noetic-gazebo-ros \
    ros-noetic-navigation ros-noetic-interactive-markers &&\
    apt install -y ros-noetic-dynamixel-sdk &&\
    apt install -y ros-noetic-turtlebot3-msgs &&\
    apt install -y ros-noetic-turtlebot3 

# Add source to .bashrc:
RUN echo 'source /root/catkin_ws/scripts/source' >> /root/.bashrc
RUN echo 'export TURTLEBOT3_MODEL=waffle_pi' >> ~/.bashrc

CMD ["bash"]
