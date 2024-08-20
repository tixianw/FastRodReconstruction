import sys
sys.path.append('../')

from reconstruction import ReconstructionModel

def main():
    # Load the model
    model = ReconstructionModel('model.pkl')

    # Create a new instance of the model
    result = model('marker_data.pkl')

    print(result)

if __name__ == '__main__':
    main()