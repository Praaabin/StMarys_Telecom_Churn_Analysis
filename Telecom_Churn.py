
import pandas as pd

class Churn_Analysis:

    def __init__(self, data_path):
        """Initializing the ChurnAnalysis class with the dataset path"""
        # Loading the dataset
        self.data = pd.read_csv(data_path, header=0, sep=',')
        print('Dataset:')
        print(self.data.head())

        # Display the number of rows in the dataset
        print('\nTotal Number of rows:', self.data.shape[0])

        # Display the number of columns in the dataset
        print('Total Number of columns:', self.data.shape[1], '\n')

        # Displaying success message
        print("Dataset loaded successfully!")


if __name__ == "__main__":
    # Creating instance of Churn_Analysis
    analysis = Churn_Analysis("Telco-Customer-Churn.csv")

