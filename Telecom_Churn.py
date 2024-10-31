import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from scipy.stats import ttest_ind


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
        print("\n     Starting Data Preprocessing:  ")
        print(f"Original structure of the dataset: {self.data.shape}")

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


    def transform_data(self):
        """Data transformation: one-hot encoding, feature engineering, normalization, and standardization."""
        print("\nStarting Data Transformation:")

        # Identifying Categorical Columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        print(f"Identified categorical columns for one-hot encoding: {list(categorical_columns)}")

        # Applying One-Hot Encoding
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Changed `sparse` to `sparse_output`
        encoded_data = pd.DataFrame(encoder.fit_transform(self.data[categorical_columns]))
        encoded_data.columns = encoder.get_feature_names_out(categorical_columns)
        self.data = pd.concat([self.data.drop(columns=categorical_columns), encoded_data], axis=1)
        print("\nOne-hot encoding completed.")

        # Printing the dataset after one-hot encoding
        print("\nDataset After One-Hot Encoding:")
        print(self.data.head())  # Shows the first few rows after one-hot encoding

        # Showing Original Descriptive Statistics before normalizing and standardizing Monthly charges and Total charges.
        columns_to_describe = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Yes']
        original_stats = self.data[columns_to_describe].describe().round(2)
        print("\nDescriptive Statistics for Original Data:")
        print(original_stats)

        # Creating new feature "TotalCharges" if it doesn't exist
        if 'TotalCharges' not in self.data.columns and 'MonthlyCharges' in self.data.columns and 'tenure' in self.data.columns:
            self.data['TotalCharges'] = self.data['MonthlyCharges'] * self.data['tenure']
            print("\nNew feature 'TotalCharges' created.")

        # Normalizing using MinMaxScaler
        minmax_scaler = MinMaxScaler()
        columns_to_scale = ['TotalCharges', 'MonthlyCharges']  # Add columns you want to normalize and standardize
        self.data[columns_to_scale] = minmax_scaler.fit_transform(self.data[columns_to_scale])
        print("\nNormalization completed for 'TotalCharges' and 'MonthlyCharges'.")

        # Standardizing using StandardScaler
        standard_scaler = StandardScaler()
        self.data[columns_to_scale] = standard_scaler.fit_transform(self.data[columns_to_scale])
        print("\nStandardization completed for 'TotalCharges' and 'MonthlyCharges'.")

        # Printing the final transformed dataset after normalization and standardization.
        print("\nshowing normalized and standardized data of Monthly charges and Total charges ")
        print(self.data[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Yes']].head())

        # Printing the final transformed dataset, after all transformations
        print("\n Transformed Dataset:")
        print(self.data.head())

        self.transformed_data = self.data.copy()
        print("\nData transformation completed!")

    def descriptive_statistics(self):
        """Calculating and printing measures of central tendency and dispersion, and visualizing data distribution."""

        # Summary statistics for the entire dataset
        print("\nDescriptive Statistics for the Transformed Dataset:")
        print(self.data.describe().round(2))  # Summary statistics for all numeric columns


        # Visualization 1: Histogram for MonthlyCharges
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data['MonthlyCharges'], kde=True, bins=30)
        plt.title('Distribution of Monthly Charges')
        plt.xlabel('Monthly Charges')
        plt.ylabel('Frequency')
        plt.show()

        # Check if Churn column exists after one-hot encoding
        churn_column = 'Churn' if 'Churn' in self.data.columns else 'Churn_Yes'

        # Visualization 2: Box plot for MonthlyCharges by Churn
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=churn_column, y='MonthlyCharges', data=self.data)
        plt.title('Monthly Charges as a Function of Churn')
        plt.xlabel('Churn')
        plt.ylabel('Monthly Charges')
        plt.show()

        print("\nDescriptive Statistics of the transformed Data completed !")

    def inferential_statistics(self):
        """Performing  hypothesis testing to determine significant differences between churned and non-churned customers."""
        print("\n  Starting Inferential Statistics: ")

        # Separating the data into churned and non-churned groups
        churned = self.data[self.data['Churn_Yes'] == 1]
        non_churned = self.data[self.data['Churn_Yes'] == 0]

        # Making List of numerical columns to test
        numerical_columns = ['MonthlyCharges', 'TotalCharges', 'tenure']

        # Performing t-tests for each numerical column
        for col in numerical_columns:
            t_stat, p_value = ttest_ind(churned[col], non_churned[col], equal_var=False)  # Welch's t-test
            print(f"\nT-Test for {col}:")
            print(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")

            # Interpreting of p-value
            if p_value < 0.05:
                print(
                    f"Result: Yes, there is a statistically significant difference in {col} between churned and non-churned customers.")
            else:
                print(
                    f"Result: No, there is not statistically significant difference in {col} between churned and non-churned customers.")

        print("\n Inferential statistics completed! ")

    def clustering(self):
        """Determining optimal clusters, applying K-means, and visualizing labeled clusters."""
        print("\n  Starting Clustering Analysis:  ")

        # Elbow Method for determining optimal k
        features = self.data[['MonthlyCharges', 'TotalCharges']]
        wcss = []  # Within-Cluster Sum of Squares
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, max_iter=5000, random_state=42)
            kmeans.fit(features)
            wcss.append(kmeans.inertia_)

        # Ploting WCSS to visualize the elbow point
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('Elbow Method for Optimal k')
        plt.show()

        # Applying K-means clustering with optimal k determined from the elbow plot
        optimal_k = 3  # Choosing from the elbow plot observation
        kmeans = KMeans(n_clusters=optimal_k, max_iter=5000, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(features)

        # Calculating means to understand cluster characteristics
        cluster_means = self.data.groupby('Cluster')[['MonthlyCharges', 'TotalCharges']].mean()
        print("\n Cluster Means (Standardized):")
        print(cluster_means)

        # Labeling clusters based on standardized mean characteristics
        # Adjusting labels based on cluster_means output
        cluster_labels = {
            0: 'Low Monthly & Low Total Charges',  # If cluster is 0, it shows lower means.
            1: 'High Monthly & High Total Charges',  # If cluster is 1, it shows higher means.
            2: 'High Monthly but Moderate Total Charges'  # If cluster is 2, shows mixed means.
        }
        self.data['Cluster'] = self.data['Cluster'].map(cluster_labels)

        # Visualizing clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='MonthlyCharges', y='TotalCharges', hue='Cluster', palette='viridis')
        plt.title('Customer segments by Monthly and Total Charges')
        plt.xlabel('Monthly Charges')
        plt.ylabel('Total Charges')
        plt.legend(title='Cluster')
        plt.show()




    def run_analysis(self):
        """ Running the complete churn analysis """
        print("\n Starting full analysis of the dataset")
        self.preprocess_data()
        self.transform_data()
        self.descriptive_statistics()
        self.inferential_statistics()
        self.clustering()





if __name__ == "__main__":
    # Creating instance of Churn_Analysis
    analysis = Churn_Analysis("Telco-Customer-Churn.csv")

    # Run full analysis
    analysis.run_analysis()

