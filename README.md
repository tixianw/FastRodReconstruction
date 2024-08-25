# Fast Rod Reconstruction

    This provides instructions for setting up and running a fast rod reconstruction on a ROS2 with Vicon system using Docker containers.
    Follow these steps to get the system up and running.

## Prerequisites

- Docker installed on your system: [ros2-vicon](https://github.com/hanson-hschang/ros2-vicon)
- Two terminal windows available
- Connect to the Vicon wifi if available
- For developers, please see the `README.md` file under `src`

## Step 1: Run the Vicon Client

### Option 1: Vicon System
If you have connected to the Vicon wifi, then in the first terminal, run the following command to start the Vicon client:
```
docker run -it --rm hansonhschang/ros2-vicon ros2 launch vicon_receiver client.launch.py
```
This command launches a Vicon client within a Docker container.

### Option 2: Mock Vicon System
If you don't have access to the Vicon, then in the first terminal, run the following command to start the Vicon mock client:
```
docker run -it --rm hansonhschang/ros2-vicon ros2 launch vicon_receiver mock_client.launch.py
```
This command launches a Vicon mock client within a Docker container.

## Step 2: Start the ROS2 Vicon Container
In the second terminal, run:
```
docker run -it --rm hansonhschang/ros2-vicon
```
This creates a container from the image and the terminal will show up like the following and gives the id `<container_id>` of the container.
In this case the id is `7cb5009f616f`.
```
root@7cb5009f616f:/#
```

## Step 3: Install this Fast Rod Reconstruction directory
Remain in the second terminal, and download this repository into the running container:
```
git clone https://github.com/tixianw/FastRodReconstruction.git
cd FastRodReconstruction
```

Now, follow the next two steps and its corresponding instruction to install Poetry and set up the package:

1. Install Poetry:
```
make install-poetry
```
> **Important**: If you see a message about adding additional directory to your `PATH`, make sure to follow the provided instructions to make this change.

2. Install the repository as a package:
```
make install
```
> **Important**: Poetry manages its own virtual environments. If you're not already in a virtual environment, Poetry will create a new one for this project. To activate this environment and use the installed package, please follow the provided instructions.

## Step 4: Launch the Reconstruction Node
Change directory to the `examples` folder:
```
cd examples
```
To launch the reconstruction node:
```
python3 launch_reconstruction_node.py
```
This command will launch the reconstruction node, which will begin listening to multiple ROS2 topics related to the Vicon system, and run the fast rod reconstruction.

## Additional Information
- The first container runs a Vicon client, which access the data provided by a Vicon motion capture system (or a mock one).
- The second container is the main ROS2 environment where you'll run your reconstruction-related nodes and scripts.
- The `launch_reconstruction_node.py` script allows you to listen to multiple ROS2 topics related to the Vicon data, and run the fast reconstruction.

## Next Steps
After launching the reconstruction node, you should see output in your terminal showing data from various reconstruction-related topics. You can now start analyzing this data or integrating it with other ROS2 nodes as needed for your specific application.

For more advanced usage, consider:
- Modifying the `launch_reconstruction_node.py` script to process the Vicon data in ways specific to your project needs.
- Creating additional ROS2 nodes that subscribe to the Vicon data and perform more complex operations.
- Integrating this Vicon data with other sensors or systems in your ROS2 environment.

Refer to the specific documentation of your Vicon receiver package for more detailed information on the data structure and available topics.
