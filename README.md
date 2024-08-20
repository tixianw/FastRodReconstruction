# FastDataSmoothing

This README provides instructions for setting up and running a fast reconstruction on a ROS2 with Vicon system using Docker containers. Follow these steps to get the system up and running.

## Prerequisites

- Docker installed on your system
- Three terminal windows available

## Step 1: Run the Vicon Client

In the first terminal, run the following command to start the Vicon client:
```
docker run -it --rm hansonhschang/ros2-vicon:latest ros2 launch vicon_receiver client.launch.py
```

This command launches a Vicon client within a Docker container.

## Step 2: Start the ROS2 Vicon Container

In the second terminal, run:
```
docker run -it --rm hansonhschang/ros2-vicon
```
This creates a container from the image and the terminal will show up like the following and gives us the id `<container_id>` of the container. 
In this case the id is `7cb5009f616f`.
```
root@7cb5009f616f:/#  
```
Once inside the container, change to the Vicon workspace directory:
```
cd vicon_ws
```

## Step 3: Copy the Reconstruction Node

In the third terminal, use the following command to copy the `reconstruction_node.py` script into the running container:
```
docker cp ./ros2-vicon/reconstruction_node.py <container_id>:/vicon_ws
```
Replace `<container_id>` with the ID of the container you noted in Step 2.

## Step 4: Run the Reconstruction Node

Return to the second terminal where you started the ROS2 Vicon container. Now that you've copied the `reconstruction_node.py` script into the container, you can run it:
```
python3 reconstruction_node.py
```

This command will start the reconstruction node, which will begin listening to multiple ROS2 topics related to the Vicon system, and run the fast reconstruction.

## Additional Information

- The first container runs a Vicon client, which access the data provided by a Vicon motion capture system.
- The second container is the main ROS2 environment where you'll run your reconstruction-related nodes and scripts.
- The `reconstruction_node.py` script is copied into the main container, allowing you to listen to multiple ROS2 topics related to the Vicon data, and run the fast reconstruction.

## Troubleshooting

- If you encounter any permission issues when copying files, you may need to run the `docker cp` command with sudo privileges.
- Ensure that the `reconstruction_node.py` file is in the directory where you're `docker cp` command is refering to.
- If you get an error when trying to run the `reconstruction_node.py` script, make sure you're in the correct directory (`vicon_ws`) and that the file was successfully copied into the container.

## Next Steps

After running the reconstruction node, you should see output in your terminal showing data from various reconstruction-related topics. You can now start analyzing this data or integrating it with other ROS2 nodes as needed for your specific application.

For more advanced usage, consider:
- Modifying the `reconstruction_node.py` script to process the Vicon data in ways specific to your project needs.
- Creating additional ROS2 nodes that subscribe to the Vicon data and perform more complex operations.
- Integrating this Vicon data with other sensors or systems in your ROS2 environment.

Refer to the specific documentation of your Vicon receiver package for more detailed information on the data structure and available topics.

