# ReMEmbR Nova Carter Demo

This directory contains the code and instructions to run ReMEmbR on a real Nova Carter robot using Isaac ROS.

## Table of Contents

- [Instructions](#instructions)
  - [Step 1 - Setup the Nova Carter](#step-1---setup-the-nova-carter)
  - [Step 2 - Build an occupancy grid map](#step-2---build-an-occupancy-grid-map)
  - [Step 3 - Run the memory builder](#step-3---run-the-memory-builder)
  - [Step 4 - Run the navigation demo](#step-4---run-the-navigation-demo)
  - [Step 5 - Integrate speech recognition](#step-5---integrate-speech-recognition)
- [Reference](#reference)
  - [Agent Node](#agent-node)
  - [Captioner Node](#captioner-node)
  - [Memory Builder Node](#memory-builder-node)
  - [ASR Node](#asr-node)

## Instructions

### Step 1 - Setup the Nova Carter

> This assumes you have a joystick controller connected, which much be done before any containers are launch to ensure the device is mounted.

1. Make workspace (we will use directory ``isaac_ros_ws`` relative to this example directory.)

    ```bash
    mkdir -p isaac_ros_ws/src
    export ISAAC_ROS_WS=$(pwd)/isaac_ros_ws
    ```

    ```bash
    cd ${ISAAC_ROS_WS}/src && \
    git clone -b release-3.1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git isaac_ros_common
    ```

2. Setup workspace (taken from [here](https://nvidia-isaac-ros.github.io/robots/nova_carter/getting_started.html#nova-carter-dev-setup)):


    ```bash
    cd ${ISAAC_ROS_WS}/src && \
      git clone --recursive https://github.com/NVIDIA-ISAAC-ROS/nova_carter.git
    ```

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common/scripts
    echo -e "CONFIG_IMAGE_KEY=ros2_humble.nova_carter\nCONFIG_DOCKER_SEARCH_DIRS=(../../nova_carter/docker ../docker)" > .isaac_ros_common-config
    ```

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common/scripts && \
      echo -e "-v /etc/nova/:/etc/nova/\n-v /opt/nvidia/nova/:/opt/nvidia/nova/" > .isaac_ros_dev-dockerargs
    ```

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

    ```bash
    cd /workspaces/isaac_ros-dev
    vcs import --recursive src < src/nova_carter/nova_carter.repos
    ```

3. Build Teleop Dependencies (taken from [here](https://nvidia-isaac-ros.github.io/robots/nova_carter/demo_teleop.html)):

    ```bash
    sudo apt update
    rosdep update
    rosdep install -i -r --from-paths ${ISAAC_ROS_WS}/src/nova_carter/nova_carter_bringup/ --rosdistro humble -y
    ```

    ```bash
    colcon build --symlink-install --packages-up-to nova_carter_bringup  --packages-skip isaac_ros_ess_models_install isaac_ros_peoplesemseg_models_install
    source install/setup.bash
    ```

4. Run teleop

    ```bash
    export ROS_DOMAIN_ID=1 

    ros2 launch nova_carter_bringup teleop.launch.py \
        enable_3d_lidar_localization:=False \
        enable_3d_lidar:=False \
        enabled_2d_lidars:=front_2d_lidar,back_2d_Lidar \
        enable_nvblox_costmap:=False \
        enabled_fisheye_cameras:=none \
        disable_nvblox:=True

    ```

5. Exit the telop ctrl+C

### Step 2 - Build an occupancy grid map

    
1. If not already inside the Isaac ROS dev container, launch the Isaac ROS Dev container (built above)

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
      ./scripts/run_dev.sh

    source /opt/ros/humble/setup.bash
    source install/setup.bash
    ```
    
1. Launch the Isaac ROS lidar mapping (ensure you shut down the previous teleop example)
    ```bash
    export ROS_DOMAIN_ID=1

    ros2 launch nova_carter_bringup lidar_mapping.launch.py \
        disable_nvblox:=True \
        enable_nvblox_costmap:=False
    ```
    
3. Open FoxGlove to visualize the map building process. (Details in Isaac Ros Documenation).

2. Teleoperate the robot to build a map

3. Save the map to a file

    ```bash
    mkdir -p ${ISAAC_ROS_WS}/maps
    ros2 run nav2_map_server map_saver_cli --fmt png -f ${ISAAC_ROS_WS}/${ISAAC_ROS_WS}/my_map.yaml
    ```

4. The map should now be stored in the ``./isaac_ros_ws/maps/my_map.yaml``

### Step 3 - Run the memory builder

#### Step 3.1 - Launch the navigation stack

1. Launch the Isaac ROS Dev container (built above)

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
      ./scripts/run_dev.sh

    source /opt/ros/humble/setup.bash
    source install/setup.bash
    ```

2. Run the navigation using the map from above.

    ```bash
    export ROS_DOMAIN_ID=1
    
    ros2 launch nova_carter_bringup navigation.launch.py \
    map_yaml_path:=/workspaces/isaac_ros-dev/maps/my_map.yaml \
    enable_3d_lidar_localization:=True \
    enable_3d_lidar:=True \
    enable_nvblox_costmap:=False \
    enabled_stereo_cameras:=none \
    enabled_fisheye_cameras:=none \
    disable_nvblox:=True

    ```

#### Step 3.1 - Launch the MilvusDB server

1. TODO

#### Step 3.2 - Launch the memory builder

1. Launch the demo container

    ```bash
    ./scripts/run_l4t_docker.sh
    ```

2. Run the memory builder node

    ```bash
    python python/memory_builder_node.py
    ```

Now, simply teleoperate the robot to populate the memory database.

### Step 4 - Run the navigation demo

#### Step 4.1 - Launch the navigation stack

1. Launch the Isaac ROS Dev container (built above)

    ```bash
    cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
      ./scripts/run_dev.sh

    source /opt/ros/humble/setup.bash
    source install/setup.bash
    ```

2. Run the navigation using the map from above.

    ```bash
    export ROS_DOMAIN_ID=1
    
    ros2 launch nova_carter_bringup navigation.launch.py \
    map_yaml_path:=/workspaces/isaac_ros-dev/maps/my_map.yaml \
    enable_3d_lidar_localization:=True \
    enable_3d_lidar:=True \
    enable_nvblox_costmap:=False \
    enabled_stereo_cameras:=none \
    enabled_fisheye_cameras:=none \
    disable_nvblox:=True

    ```

#### Step 4.2 - Launch the ReMEmbR Agent node and test

1. Launch the demo container

    ```bash
    ./scripts/run_l4t_docker.sh
    ```

2. Launch the ReMEmbR Agent node

    ```bash
    python python/agent_node.py
    ```

3. Send a test query

    ```bash
    ros2 topic pub /speech std_msgs/String "data: Hey robot, can you take me to get some snacks?"
    ```


### Step 5 - Integrate speech recognition

1. Connect a microphone to the Nova Carter via USB

> This is tested with a Respeaker microphone connected.  You may need to modify the device index to your microphone.

2. Make directory to cache speech recognition enginers

    ```bash
    mkdir -p data/asr
    ```

3. Attach to the demo container

    ```bash
    docker exec -it nova_carter_demo bash
    ```

3. Run the speech recongition Node.  

    ```bash
    python3 python/asr_node.py
    ```

Now, you should be able to talk to the robot and see speech!  

Assuming the agent node is still running, these queries are forwarded to the robot.

> Note, queries are filtered for the keyword "robot".

## Reference

Below are details about the ROS nodes used in the demo.  You can check the python folder for additional details.

### Agent Node

| Name | Description | Default |
|------|-------------|---------|
| llm_type | The LLM model to use for the ReMEmbR agent. | "command-r" |
| db_collection | The MilvusDB collection to use for the memory | "test_collection" |
| db_ip | The MilvusDB IP address. | "127.0.0.1" |
| query_topic | The topic to listen to queries from. | "/speech" |
| pose_topic | The topic to listen to current robot poses from. | "/amcl_pose" |
| goal_pose_topic | The topic to publish goal poses to. | "/goal_pose" |


### Captioner Node

| Name | Description | Default |
|------|-------------|---------|
| model | The VILA model to use for captioning. | "Efficient-Large-Model/VILA1.5-3B" |
| segment_duration | The time window (in seconds) to caption. | 3 |
| image_topic | The topic to subscribe to for images to caption. | "/front_stereo_camera/left/image_raw" |
| caption_topic | The topic to publish captions to. | "/caption" |

### Memory Builder Node

| Name | Description | Default |
|------|-------------|---------|
| db_collection | The collection name in MilvusDB to add entries. | "test_collection" |
| db_ip | The MilvusDB IP address. | "127.0.0.1" |
| pose_topic | The topic to subscribe to get pose information. | "/amcl_pose" | 
| caption_topic | The topic to subscribe to get captions. | "/caption" |

### ASR Node

| Name | Description | Default |
|------|-------------|---------|
| model | The Whisper model to use. | "small.en" |
| backend | The Whisper backend to use. | "whisper_trt" |
| cache_dir | Directory to cache the built models. | None |
| vad_window | Number of audio chunks to use in max-filter window for voice activity detection. | 5 |
| mic_device_index | The microphone device index. | None |
| mic_sample_rate | The microphone sample rate. | 16000 |
| mic_channels | The microphone number of channels. | 6 |
| mic_bitwidth | The microphone bitwidth. | 2 |
| speech_topic | The topic to publish speech segments to. | "/speech" |
