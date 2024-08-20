from reconstruction import ReconstructionResult

class ReconstructionModel:
    def __init__(self, file_name: str):
        self.file_name = file_name
        # load the model from file


        self.number_of_elements = 100 # update this value from the loaded model

        self.result = ReconstructionResult(self.number_of_elements)

    def __call__(self, marker_data) -> ReconstructionResult:
        # update the result with the new marker data

        return self.result
