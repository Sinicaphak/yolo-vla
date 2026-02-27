cd /home/apollo/disk/ros2/src
pip install casadi
sudo apt install ros-humble-osqp-vendor
git clone https://github.com/agilexrobotics/ugv_sdk.git
git clone https://github.com/agilexrobotics/hunter_ros2.git
cd ..
colcon build


sudo apt install v4l-utils
v4l2-ctl --list-devices    
nvgstcapture-1.0 --camsrc=0 --cap-dev-node=1

ros2/tools/jetson-gs_usb-kernel-builder.sh 这个脚本编译并安装gs_usb驱动
orin的usb转can的端口号是can0, ros2/src/ugv_sdk/scripts里面要改一下

停止包中的所有节点
pkill -f car &&
pkill -f all_launcher &&
pkill -f mpc_planner &&
ros2 node list
