import numpy as np
import pandas as pd

class Churn_Analysis:

    def __init__(self, data_path):
        """Initializing the ChurnAnalysis class with the dataset path"""
        # Loading the dataset
        self.data = pd.read_csv(data_path, header=0, sep=',')
        print('Dataset:')
        print(self.data.head())

        # Displaying the total number of rows in the dataset
        print('\nTotal Number of rows:', self.data.shape[0])

        # Displaying the number of columns in the dataset
        print('Total Number of columns:', self.data.shape[1], '\n')

        # Displaying success message
        print("Dataset loaded successfully!")

    def preprocess_data(self):
        """ Comprehensive data preprocessing and cleaning """
        print("\n=== Starting Data Preprocessing ===")
        print(f"Initial shape of the dataset: {self.data.shape}")

        # 1. Remove duplicates
        num_duplicates = self.data.duplicated().sum()
        self.data.drop_duplicates(inplace=True)
        print(f"\nNumber of duplicates in the dataset: {num_duplicates}")
        print(f"Shape after removing duplicates from the dataset: {self.data.shape}")

        # 2. Handle missing values
        self.data.replace(['', 'None'], np.nan, inplace=True)
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
        self.data['tenure'] = pd.to_numeric(self.data['tenure'], errors='coerce')

        # Print missing values before handling
        missing_values_per_column = self.data.isnull().sum()
        total_missing_values = missing_values_per_column.sum()
        print("\nMissing values per column:")
        print(missing_values_per_column)
        print(f"Total number of missing values: {total_missing_values}")

        # Impute missing values
        numerical_columns = ['MonthlyCharges', 'TotalCharges']
        for col in numerical_columns:
            self.data[col] =  self.data[col].fillna(self.data[col].mean())

        categorical_columns =  self.data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

        # Print missing values after handling
        missing_values_after_handling = self.data.isnull().sum()
        print("\nMissing values after handling:")
        print(missing_values_after_handling)

        # 3. Handle outliers
        def handle_outliers(df, column):
            Q1 =  df[column].quantile(0.25)
            Q3 =  df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[column].clip(lower=lower_bound, upper=upper_bound)

        num_outliers = 0
        for col in numerical_columns:
            before_outlier_count = self.data[col].isnull().count()
            self.data[col] = handle_outliers(self.data, col)
            after_outlier_count = self.data[col].isnull().count()
            num_outliers += before_outlier_count - after_outlier_count

        print(f"\nNumber of outliers handled: {num_outliers}")

        # Print shape after handling outliers
        print(f"Shape after handling outliers: {self.data.shape}")

        # Final data summary
        print("\n=== Final Data Summary ===")
        print(self.data.describe(include='all'))

        self.preprocessed_data = self.data.copy()
        print("\nPreprocessing completed!")
        print(f"Final shape of the dataset: {self.preprocessed_data.shape}")

    def run_analysis(self):
        """ Running the complete churn analysis """
        print("\n Starting full analysis of the dataset")
        self.preprocess_data()




if __name__ == "__main__":
    # Creating instance of Churn_Analysis
    analysis = Churn_Analysis("Telco-Customer-Churn.csv")

    # Run full analysis
    analysis.run_analysis()

