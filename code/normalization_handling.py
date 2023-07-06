import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_dataset(dataset):
    # Convert the dataset to a NumPy array
    dataset = np.array(dataset)
    
    # Split the features and labels
    features = dataset[:, :-1]
    labels = dataset[:, -1]
    
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Normalize the features using min-max scaling
    normalized_features = scaler.fit_transform(features)
    
    # Replace missing values with the mean value of each feature
    mean_values = np.nanmean(normalized_features, axis=0)
    normalized_features = np.where(np.isnan(normalized_features), mean_values, normalized_features)
    
    # Combine the normalized features and labels
    normalized_dataset = np.column_stack((normalized_features, labels))
    
    return normalized_dataset