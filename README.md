# Dataset

The dataset consists of images from 5 different dog breeds. The training and validation data will be obtained through the Kaggle API. The API's functionality to download only metadata will be utilized to check for updates in the dataset, ensuring that the most current data is used.

# Project Explanation

The goal of this project is to develop a machine learning model capable of classifying images into one of five dog breeds. 

# Flows & Actions

## Collect Flow

Tasks involved in the Collect flow:

1. **Load Dataset Metadata**: Utilize the Kaggle API to retrieve metadata of the dataset.
2. **Check Dataset Status**: Verify if the dataset exists locally and whether the local version is outdated compared to the version on Kaggle.
3. **Download Dataset**: Load the dataset from Kaggle onto the local disk.
4. **Extract Dataset**: Unpack the dataset files ready for preprocessing.

## Prep Flow

Tasks involved in the Prep flow:

1. **Read Data**: Load the dataset from the directory where it was extracted.
2. **Cache Data**: Cache the dataset to optimize loading times for TensorFlow processing.

## HPO (Hyper Parameter Optimization) Flow

Tasks involved in the HPO flow:

1. **Perform HPO**: Execute hyperparameter optimization to enhance model performance.

## Register Flow

Tasks involved in the Register flow:

1. **Train and Log Model**: Conduct model training while logging performance metrics.
2. **Experiment Runs**: Collect data from various training runs.
3. **Model Selection**: Choose the best performing model based on the metrics.
