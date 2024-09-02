import time
from collections import OrderedDict

from ros2_vicon.node import LoggerNode


class StageNode(LoggerNode):

    def __init__(self, ordered_stages: OrderedDict, **kwargs):
        super().__init__(**kwargs)
        self.ordered_stages = ordered_stages
        self.stages = list(ordered_stages.keys())
        self.current_stage = None
        self.__stage_time = time.time()

    def next_stage(self):
        current_index = self.stages.index(self.current_stage)
        if current_index + 1 < len(self.stages):
            self.log_info(
                f"\tStage {self.current_stage} running... Done. "
                f"Elapsed time: {time.time() - self.__stage_time} [sec]."
            )
            self.__stage_time = time.time()
            self.current_stage = self.stages[current_index + 1]
            self.log_info(f"\tStage {self.current_stage} running...")
        else:
            self.log_warn("\tAll stages are done.")

    def stage(self) -> bool:
        if self.current_stage is None:
            self.__stage_time = time.time()
            self.current_stage = self.stages[0]
            self.log_info(f"\t")
            self.log_info(f"\tStage {self.current_stage} running...")
        return self.ordered_stages[self.current_stage]()
