import sys
sys.path.append('../')

import numpy as np

from reconstruction import ReconstructionModel

def main():
    # Load the model
    model = ReconstructionModel('model.pt')

    marker_data = np.zeros(45)

    # Create a new instance of the model
    result = model(marker_data)

    print(result)

if __name__ == '__main__':
    main()