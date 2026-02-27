from launch import LaunchDescription
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch_ros.actions import Node

# 图片目录
PIC_DIR = "/home/apollo/disk/ros2/src/car/pic/8"
# 话题
PIC_TOPIC = "/car/pic"
PROCESS_PIC_TOPIC = "/car/process_pic"
COMMD_TOPIC = "/goal_point"
MODE="local"
# MODE="camera"
# 发布频率(fps)
FPS = 1
# 模型API
API_URL = "http://localhost:8003/v1/chat/completions"
# 图片压缩质量 (1-100, 越低压缩率越高)
COMPRESSION_QUALITY = 30
IMG_WIDTH=426
IMG_HIGHT=240
# 模型输出最大token数
MAX_TOKENS = 100

TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 5

# prompt
TEXT_1="""
You are an autonomous driving planner.
Coordinate system: X-axis is lateral, Y-axis is longitudinal.
The ego vehicle is at (0,0), units are meters.
Based on the provided front-view image and driving context, plan future waypoints at 0.5-second intervals for the next 3 seconds.

Here is the front-view image from the car:
"""

TEXT_2="""
Mission goal: TURN LEFT
Traffic rules:
- Avoid collision with other objects.
- Always drive on drivable regions.
- Avoid occupied regions.

Please plan future waypoints at 0.5-second intervals for the next 3 seconds.
"""



def generate_launch_description():
    log_level_arg = DeclareLaunchArgument(
        'log_level', default_value='info',
        description='Logger level for all nodes')

    log_level = LaunchConfiguration('log_level')
    
    return LaunchDescription([
        Node(
            package='car',
            executable='vllm_ask',
            name='vllm_ask',
            parameters=[{
                'pic_topic': PIC_TOPIC,
                'process_pic_topic': PROCESS_PIC_TOPIC,
                'commd_topic': COMMD_TOPIC,
                'api_url': API_URL,
                'compression_quality': COMPRESSION_QUALITY,
                'img_width': IMG_WIDTH,
                'img_hight': IMG_HIGHT,
                'max_tokens': MAX_TOKENS,
                'text_1': TEXT_1,
                'text_2': TEXT_2,
                'temperature': TEMPERATURE,
                'top_p': TOP_P,
                'top_k': TOP_K,
            }],
            arguments=['--ros-args', '--log-level', log_level],
        ),
        Node(
            package='car',
            executable='image_publisher',
            name='image_publisher',
            parameters=[{
                'pic_topic': PIC_TOPIC,
                'fps': FPS,
                'pic_dir': PIC_DIR,
                'mode': MODE,
                'camera_device': '/dev/video0',
            }],
            arguments=['--ros-args', '--log-level', log_level],
        ),
    ])
    
