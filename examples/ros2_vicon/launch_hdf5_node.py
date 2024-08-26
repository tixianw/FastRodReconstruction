from typing import Dict

from dataclasses import dataclass

import click

import ros2_vicon
from ros2_vicon import HDF5Writer, NDArraySubscriber, PoseSubscriber
from ros2_vicon.message.array import NDArrayMessage
from ros2_vicon.message.pose import PoseMessage
from ros2_vicon.node import LoggerNode


@dataclass
class SubscriptionInfo:
    topic: str
    message: NDArrayMessage


@dataclass
class WriterInfo:
    file_name: str
    chunk_size: int
    writing_rate: float = 10.0


class HDF5WriterNode(LoggerNode):
    def __init__(
        self,
        subscriptions_info: Dict[str, SubscriptionInfo],
        writer_info: WriterInfo,
        log_level: str = "info",
    ):
        super().__init__(
            node_name="hdf5_writer_node",
            log_level=log_level,
        )

        self.subscriptions_info = subscriptions_info
        self.writer_info = writer_info

        # Initialize subscribers
        self.log_info("- Subcribers initializing...")
        self.__subscribers = {
            key: NDArraySubscriber(
                message=subscription_info.message,
                topic=subscription_info.topic,
                callback=self.subscriber_callback_closure(key),
                qos_profile=100,
                node=self,
            )
            for key, subscription_info in self.subscriptions_info.items()
        }

        # Initialize hdf5 writer
        self.log_info("- HDF5 writer initializing...")
        self.writer = HDF5Writer(
            file_name=self.writer_info.file_name,
            messages={
                key: subscription_info.message
                for key, subscription_info in self.subscriptions_info.items()
            },
            node=self,
            chunk_size=self.writer_info.chunk_size,
            writer_period_sec=1.0 / self.writer_info.writing_rate,
        )
        self.ready()
        self.writer.start()

    def subscriber_callback_closure(self, key: str) -> callable:
        def subscriber_callback(msg):
            self.log_debug(f"{self.__subscribers[key]}")
            self.__subscribers[key].receive(msg)
            self.writer.record(key, msg)

        return subscriber_callback

    def stop(self):
        self.writer.stop()
        super().stop()


@click.command()
@click.option(
    "--log-level",
    type=click.Choice(
        ["debug", "info", "warn", "error", "fatal"],
        case_sensitive=False,
    ),
    default="info",
    help="Set the logging level",
)
@click.option(
    "--number-of-markers",
    type=int,
    default=4,
    help="Set the number of markers",
    callback=lambda ctx, param, value: (
        value if value >= 1 else ctx.fail("Must be greater than or equal to 1")
    ),
)
@click.option(
    "--number-of-elements",
    type=int,
    default=100,
    help="Set the number of elements",
    callback=lambda ctx, param, value: (
        value if value >= 1 else ctx.fail("Must be greater than or equal to 1")
    ),
)
@click.option(
    "--writing-rate",
    type=float,
    default=10,
    help="Set the writing rate",
    callback=lambda ctx, param, value: (
        value if value >= 1 else ctx.fail("Must be greater than or equal to 1")
    ),
)
@click.option(
    "--chunk-size",
    type=int,
    default=10,
    help="Set the chunk size",
    callback=lambda ctx, param, value: (
        value if value >= 1 else ctx.fail("Must be greater than or equal to 1")
    ),
)
@click.option(
    "--file-name",
    type=str,
    default="reconstructin.h5",
    help="Set the file name of the HDF5 file",
)
def main(
    log_level: str,
    number_of_markers: int,
    number_of_elements: int,
    writing_rate: float,
    chunk_size: int,
    file_name: str,
):

    ros2_vicon.init()

    subscriptions_info = {
        "pose": SubscriptionInfo(
            topic="/vicon/pose",
            message=PoseMessage(
                shape=(number_of_markers - 1,),
                axis_labels=("element",),
            ),
        ),
        "position": SubscriptionInfo(
            topic="/reconstruction/position",
            message=NDArrayMessage(
                shape=(3, number_of_elements + 1),
                axis_labels=("position", "element"),
            ),
        ),
        "directors": SubscriptionInfo(
            topic="/reconstruction/directors",
            message=NDArrayMessage(
                shape=(3, 3, number_of_elements),
                axis_labels=("directors", "director_index", "element"),
            ),
        ),
        "kappa": SubscriptionInfo(
            topic="/reconstruction/kappa",
            message=NDArrayMessage(
                shape=(3, number_of_elements - 1),
                axis_labels=("kappa", "element"),
            ),
        ),
    }
    writer_info = WriterInfo(
        file_name=file_name,
        chunk_size=chunk_size,
        writing_rate=writing_rate,
    )
    node = HDF5WriterNode(
        subscriptions_info=subscriptions_info,
        writer_info=writer_info,
        log_level=log_level,
    )

    try:
        node.start()
    except KeyboardInterrupt:
        node.stop()
        ros2_vicon.shutdown()


if __name__ == "__main__":
    main()
