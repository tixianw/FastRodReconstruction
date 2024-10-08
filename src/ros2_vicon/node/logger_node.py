import time

try:
    import rclpy
    from rclpy.node import Node
except ModuleNotFoundError:
    print(
        "Could not import ROS2 modules. Make sure to source ROS2 workspace first."
    )
    import sys

    sys.exit(1)


class LoggerNode(Node):
    def __init__(self, node_name: str, log_level: str = "info"):
        super().__init__(node_name)
        self.node_name = node_name
        self.set_logger_level(log_level)
        self.log_info(f"{node_name} initializing...")
        self.log_info(f"logger level set to {log_level}")

    def ready(self):
        self.log_info(f"{self.get_name()} is ready.")
        self.log_info(f"start in 3 seconds...")
        time.sleep(1)
        self.log_info(f"start in 2 seconds...")
        time.sleep(1)
        self.log_info(f"start in 1 second...")
        time.sleep(1)
        self.log_info(f"{self.get_name()} has started.")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.node_name})"

    def __repr__(self) -> str:
        return self.__str__()

    def set_logger_level(self, log_level: str) -> None:
        level_map = {
            "debug": rclpy.logging.LoggingSeverity.DEBUG,
            "info": rclpy.logging.LoggingSeverity.INFO,
            "warn": rclpy.logging.LoggingSeverity.WARN,
            "error": rclpy.logging.LoggingSeverity.ERROR,
            "fatal": rclpy.logging.LoggingSeverity.FATAL,
        }
        assert (
            log_level.lower() in level_map.keys()
        ), f"Invalid log level: {log_level}"
        level = level_map[log_level.lower()]
        self.get_logger().set_level(level)

    def get_logger_level(self) -> str:
        return self.get_logger().get_effective_level().name.lower()

    def log_debug(self, message: str) -> None:
        self.get_logger().debug(message)

    def log_info(self, message: str) -> None:
        self.get_logger().info(message)

    def log_warn(self, message: str) -> None:
        self.get_logger().warn(message)

    def log_error(self, message: str) -> None:
        self.get_logger().error(message)

    def log_fatal(self, message: str) -> None:
        self.get_logger().fatal(message)

    def start(self) -> None:
        rclpy.spin(self)

    def stop(self) -> None:
        self.log_info(f"{self.get_name()} shutting down...")
        super().destroy_node()

    def create_subscriber(
        self,
        msg_type,
        topic,
        callback,
        qos_profile,
    ):
        return super().create_subscription(
            msg_type=msg_type,
            topic=topic,
            callback=callback,
            qos_profile=qos_profile,
        )

    def create_publisher(
        self,
        msg_type,
        topic,
        qos_profile,
    ):
        return super().create_publisher(
            msg_type=msg_type,
            topic=topic,
            qos_profile=qos_profile,
        )
