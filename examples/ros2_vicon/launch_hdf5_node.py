from typing import Dict, Union

import time
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
    timesignal: bool = True


@dataclass
class WriterInfo:
    file_name: str
    chunk_size: int
    writing_rate: float = 10.0
    verbose: bool = False


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
                if subscription_info.timesignal
            },
            node=self,
            chunk_size=self.writer_info.chunk_size,
            writer_period_sec=1.0 / self.writer_info.writing_rate,
            verbose=self.writer_info.verbose,
        )
        self.ready()
        self.writer.start()

    def subscriber_callback_closure(self, key: str) -> callable:
        def subscriber_callback(msg):
            self.__subscribers[key].receive(msg)
            self.log_debug(f"{self.__subscribers[key]}")
            self.writer.record(key, msg)

        return subscriber_callback

    def stop(self):
        self.writer.stop()
        super().stop()


def not_smaller_then_one_constraint(
    ctx: click.Context,
    param: click.Option,
    value: Union[float, int],
) -> float:
    if value < 1:
        raise click.BadParameter("Must be greater than or equal to 1")
    return value


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
    callback=not_smaller_then_one_constraint,
)
@click.option(
    "--writing-rate",
    type=float,
    default=10,
    help="Set the writing rate",
    callback=not_smaller_then_one_constraint,
)
@click.option(
    "--chunk-size",
    type=int,
    default=100,
    help="Set the chunk size",
    callback=not_smaller_then_one_constraint,
)
@click.option(
    "--file-name",
    type=str,
    default="reconstruction.h5",
    help="Set the file name of the HDF5 file",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=True,
    help="Enable verbose output",
)
def main(
    log_level: str,
    number_of_markers: int,
    number_of_elements: int,
    writing_rate: float,
    chunk_size: int,
    file_name: str,
    verbose: bool,
):
    ros2_vicon.init()

    subscriptions_info = {
        "time": SubscriptionInfo(
            topic="/time",
            message=NDArrayMessage(
                shape=(1,),
                axis_labels=("time",),
            ),
        ),
        "pose": SubscriptionInfo(
            topic="/vicon/pose",
            message=PoseMessage(
                shape=(number_of_markers,),
                axis_labels=("marker",),
            ),
        ),
        "filtered_pose": SubscriptionInfo(
            topic="/filter/pose",
            message=PoseMessage(
                shape=(number_of_markers,),
                axis_labels=("marker",),
            ),
        ),
        "lab_frame_transformation": SubscriptionInfo(
            topic="/reconstruction/lab_frame_transformation",
            message=NDArrayMessage(
                shape=(4, 4),
                axis_labels=("transformation_matrix", ""),
            ),
        ),
        "material_frame_transformation": SubscriptionInfo(
            topic="/reconstruction/material_frame_transformation",
            message=NDArrayMessage(
                shape=(4, 4, number_of_markers),
                axis_labels=("transformation_matrix", "", "marker"),
            ),
        ),
        "calibrated_pose": SubscriptionInfo(
            topic="/reconstruction/calibrated_pose",
            message=PoseMessage(
                shape=(number_of_markers,),
                axis_labels=("marker",),
            ),
        ),
        "input": SubscriptionInfo(
            topic="/reconstruction/input",
            message=NDArrayMessage(
                shape=(4, 4, number_of_markers - 1),
                axis_labels=("input", "", "marker"),
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
        "cost": SubscriptionInfo(
            topic="/reconstruction/cost",
            message=NDArrayMessage(
                shape=(1,),
                axis_labels=("cost",),
            ),
        ),
    }
    writer_info = WriterInfo(
        file_name=file_name,
        chunk_size=chunk_size,
        writing_rate=writing_rate,
        verbose=verbose,
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
