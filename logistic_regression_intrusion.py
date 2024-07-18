# Import necessary libraries

from ucimlrepo import fetch_ucirepo
import pandas as pd
import warnings
import numpy as np
import logging
import sys


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import resample
from sklearn.metrics import r2_score
from datetime import datetime


# Define a function to fetch the dataset and create a data frame

def create_df(retries=1):
    attempt = 1
    while attempt <= retries:
        try:
            # Fetching Data Message
            print("\nFetching Data from the repository...")

            # Fetch the dataset with the specified ID
            rt_iot2022 = fetch_ucirepo(id=942)

            # Assigning X and Y into variables (as pandas dataframes) 
            X = rt_iot2022.data.features 
            y = rt_iot2022.data.targets 

            # Progress Done Message
            print("Data fetching completed successfully!")
            
            return X, y

        except Exception as e:
            print(f"Error fetching data (Attempt {attempt}/{retries}): {e}")
            attempt += 1

    print("Failed to fetch data after multiple attempts check network connection and try again.")
    sys.exit()
    return None, None

# Function to carry out all the data pre-processing

def data_preprocess(X, y):

    # Creating Data Frame Message
    print("Creating Data Frame from fetched data...")

    # Creating DataFrame from passed X and y 
    df = pd.DataFrame(X)
    df['Attack_type'] = y

    # Data Preprocessing Message
    print("Performing Data Pre-processing...")

    # Adding a Binary Column as the Target Based on Dataset Information (1=Attack, 0=Normal Traffic)
    # List of attack types to categorize as 1
    target_attacks = ['DOS_SYN_Hping', 'ARP_poisoning', 'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN',
                      'NMAP_OS_DETECTION', 'NMAP_TCP_scan', 'DDOS_Slowloris',
                      'Metasploit_Brute_Force_SSH', 'NMAP_FIN_SCAN']

    # Creating a new column with 1 for specified attack types and 0 for others
    df['target_attack'] = df['Attack_type'].isin(target_attacks).astype(int)

    # Dropping the original 'Attack_type' column
    df.drop(columns=['Attack_type'], inplace=True)

    # Categorical columns to be encoded
    categorical_cols = ['proto', 'service']

    # One-Hot Encoding the categorical columns
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(df[categorical_cols])

    # Retrieving column names for one-hot encoded features
    onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)

    # Creating a new DataFrame with one-hot encoded columns
    onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_feature_names)

    # Combine original DataFrame with one-hot encoded DataFrame
    df = df.drop(columns=categorical_cols)

    combined_df = pd.concat([df, onehot_df], axis=1)
    
    # Dropping duplicates
    combined_df = combined_df.drop_duplicates()

    return combined_df,df

# Function to undersample the dataset (used only for experiments)

def data_resample(data):

    # Resampling Data to Minimize Imbalance

    # Separating majority and minority classes
    majority_class = data[data['target_attack'] == 1]
    minority_class = data[data['target_attack'] == 0]

    # Undersampling majority class
    undersampled_majority = resample(majority_class, 
                                    replace=False,    
                                    n_samples=len(minority_class),    
                                    random_state=42)  
    
    # Combining minority class with undersampled majority class
    undersampled_df = pd.concat([undersampled_majority, minority_class])

    # Shuffling the undersampled dataset
    undersampled_df = undersampled_df.sample(frac=1, random_state=42)
    
    # Dropping the duplicates
    undersampled_df = undersampled_df.drop_duplicates()

    return undersampled_df

# Function to train the models

def train_models(combined_df):

    # Checking  or duplicates and remove if any
    if combined_df.duplicated().any():
        print("Duplicates found in the dataset.")
        combined_df = combined_df.drop_duplicates()
        print("Duplicates Removed!.")

    # Separating features and target variable
    X = combined_df.drop(columns=['target_attack'])
    y = combined_df['target_attack']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print target variable distribution in training set
    print("----------------------------Target Variable Distribution-----------------------------------")
    print("Training Set - Target Variable Distribution:")
    print(y_train.value_counts(normalize=True))

    # Print target variable distribution in test set
    print("\nTest Set - Target Variable Distribution:")
    print(y_test.value_counts(normalize=True))
    print("--------------------------------------------------------")

    print("Training set size:", y_train.shape[0])
    print("Test set size:", y_test.shape[0])

    # Standardizng features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # max_iter as set to 100 in order to run the models faster when excecution, but higher iteration were used to gain results
    max_iter = 100

    # Defining a dictionary to store models and data
    models_data = {}

    warnings.filterwarnings("ignore")

    print('----------------------Training Models-------------------------')

    # Model 1
    print("Training Model 1 (No regularization - lbfgs )...")
    lr_lbfgs = LogisticRegression(penalty=None,max_iter=max_iter)
    lr_lbfgs.fit(X_train, y_train)
    models_data['No regularization (lbfgs)'] = {
        'model': lr_lbfgs,
        'X_test_scaled': X_test,
        'y_test': y_test
    }
    print("Model 1 trained successfully!")

    # Model 2
    print("Training Model 2 (No regularization - saga)...")
    lr_saga = LogisticRegression(penalty=None,max_iter=max_iter, solver='saga')
    lr_saga.fit(X_train, y_train)
    models_data['No regularization (saga)'] = {
        'model': lr_saga,
        'X_test_scaled': X_test,
        'y_test': y_test
    }
    print("Model 2 trained successfully!")

    # Model 3
    print("Training Model 3 (L2 regularization - lbfgs)...")
    lr_l2_lbgfs_reg = LogisticRegression(penalty='l2', max_iter=max_iter, solver='lbfgs')
    lr_l2_lbgfs_reg.fit(X_train, y_train)
    models_data['L2 regularization (lbfgs)'] = {
        'model': lr_l2_lbgfs_reg,
        'X_test_scaled': X_test,
        'y_test': y_test
    }
    print("Model 3 trained successfully!")

    # Model 4
    print("Training Model 4 (L2 regularization - saga)...")
    lr_l2_saga_reg = LogisticRegression(penalty='l2', max_iter=max_iter, solver='saga')
    lr_l2_saga_reg.fit(X_train_scaled, y_train)
    models_data['L2 regularization (saga)'] = {
        'model': lr_l2_saga_reg,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }
    print("Model 4 trained successfully!") 

    # Model 5
    print("Training Model 5 (L1 (Lasso) - saga)...")
    lr_l1_reg = LogisticRegression(penalty='l1', max_iter=max_iter, solver='saga')
    lr_l1_reg.fit(X_train_scaled, y_train)
    models_data['L1 regularization (saga)'] = {
        'model': lr_l1_reg,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }
    print("Model 5 trained successfully!") 

    # Model 6
    print("Training Model 6 (L1-L2 (elastic-net) - saga)...")
    lr_ela_reg = LogisticRegression(penalty='elasticnet', l1_ratio=0.5 , max_iter=max_iter, solver='saga')
    lr_ela_reg.fit(X_train_scaled, y_train)
    models_data['L1-L2 regularization (saga)'] = {
        'model': lr_ela_reg,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }
    print("Model 6 trained successfully!") 

    return models_data

# Function to train the PCA Model

def train_pca_models(data):

    print("Training Model 7 (PCA Analysis)...")
        
    # Separate features and target variable
    X = data.drop(columns=['target_attack'])
    y = data['target_attack']

    pca_models = {}

    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Applying PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Performing PCA on sclaed X
    pca = PCA().fit(X_scaled)

    # Calculating cumulative variance explained by each component
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Finding the index where cumulative variance hits 95%
    components_needed = np.argmax(cumulative_variance >= 0.95) + 1  

    # Printing output
    print("-"*125)
    print(f" \nNumber of components needed to explain 95% variance: {components_needed}\n")
    print("-"*125)

    # Performing PCA with the optimal number of components
    pca = PCA(n_components=components_needed)
    X_pca = pca.fit_transform(X_scaled)

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y,random_state=42)

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Training PCA logistic regression model
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = logistic_model.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)

    pca_models[f'PCA Model - No of Components {components_needed}'] = {
        'model': logistic_model,
        'X_test_pca': X_test,
        'y_test': y_test,
        'components_needed': components_needed,
        'accuracy': accuracy
    }

    print("All Models trained successfully!")

    return pca_models


# Function to calculate the r_squared    

def calculate_r_squared(model_data):

    model = model_data['model']
    X_test_scaled = model_data.get('X_test_scaled')
    y_test = model_data.get('y_test')

    if X_test_scaled is not None and y_test is not None:
        # Predict on the test set
        y_pred = model.predict(X_test_scaled)
        
        # Calculate R-squared value
        r_squared = r2_score(y_test, y_pred)
        
        return r_squared
    else:
        return None
    
# Function to display the results    

def display_results(models_data):

    print("Generating Results...")

    best_model = None
    best_pca = None
    best_score = 0
    best_pca_score = 0

    print("{:<60} {:<10} {:<10} {:<10} {:<10} {:<10}".format("Model", "Accuracy", "Precision", "Recall", "F1-Score", "Explained Variance"))
    print("-" * 125)

    for model_name, data in models_data.items():
        if 'X_test_scaled' in data and 'components_needed' not in data:  # Regular models
            model = data['model']
            X_test_scaled = data['X_test_scaled']
            y_test = data['y_test']

            # Predict on test data
            predictions = model.predict(X_test_scaled)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            # Calculate R-squared
            r_squared = calculate_r_squared(data)

            # Print metrics
            print("{:<60} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}".format(model_name, accuracy, precision, recall, f1, r_squared))

            # Update best model
            if accuracy > best_score:
                best_model = model_name
                best_score = accuracy

    # Print best model based on accuracy
    print("-" * 125)
    print(f"Best Model: {best_model} (Accuracy: {best_score:.4f})")
    print("-" * 125)

    # Print PCA models
    print("\nPCA Model:")
    print("{:<60} {:<10} {:<10} {:<10} {:<10} {:<10}".format("Model", "Accuracy", "Precision", "Recall", "F1-Score", "Components"))
    print("-" * 125)

    # Iterate over all models to print PCA results and update best PCA model
    for model_name, data in models_data.items():
        if 'X_test_scaled' not in data and 'components_needed' in data:  # PCA models
            pca_model = data['model']
            X_test_pca = data['X_test_pca']
            y_test = data['y_test']
            components = data['components_needed']

            # Predict on test data
            predictions = pca_model.predict(X_test_pca)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            # Print metrics
            print("{:<60} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}".format(model_name, accuracy, precision, recall, f1, components))

            # Update best PCA model
            if accuracy > best_pca_score:
                best_pca = model_name
                best_pca_score = accuracy

    # Print best PCA model based on accuracy
    print("-" * 125)
    print(f"Best Model PCA: {best_pca} (Accuracy: {best_pca_score:.4f})")
    print("-" * 125)

    print("Note: Please note that max_iter for this version of the code is set to 100 for smoother runtime\n      But higher iterations were used to obtain the results reported.\n")


def main():

    print("\n---------CSCI933 Assignment 01------ \n   Manura Dheerasekara - 7913692")

    # Obtaining X and y with the function

    X,y = create_df()

    # X and y being passed in to data_preprocess() and a dataframe is returned

    combined_df,df = data_preprocess(X,y)

    # Only used for testing with undersampled data

    small_data = data_resample(combined_df)

    # Passing the pre-processed data into train_models(),train_pca_models() functions for training

    trained_models = train_models(combined_df)
    pca_models = train_pca_models(combined_df)
    
    # Combining all trained models 

    trained_models.update(pca_models)

    # Finally Printing the output result

    display_results(trained_models)
    
if __name__ == "__main__":
    main()
