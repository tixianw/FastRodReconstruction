try:
    import rclpy
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


class QoSProfile(rclpy.qos.QoSProfile):
    pass
