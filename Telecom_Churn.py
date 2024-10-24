import pandas as pd

import pandas as pd

class ChurnAnalysis:

    def __init__(self, data_path):
        """Initialize the ChurnAnalysis class with the dataset path"""
        # Load the dataset
        self.data = pd.read_csv(data_path, header=0, sep=',')

        # Display success message
        print("Dataset loaded successfully!")
        print('Dataset:')
        print(self.data.head())

        #Display the number of rows in the dataset
        print('\nNumber of rows:', self.data.shape[0], '\n')

if __name__ == "__main__":
    # Create instance of ChurnAnalysis
    analysis = ChurnAnalysis("Telco-Customer-Churn.csv")

