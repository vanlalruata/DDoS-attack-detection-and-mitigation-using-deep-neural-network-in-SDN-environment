import pandas as pd

def save_dataset_to_csv(training_set, testing_set, training_file, testing_file):
    # Convert the training set and testing set to pandas DataFrames
    training_df = pd.DataFrame(training_set)
    testing_df = pd.DataFrame(testing_set)
    
    # Save the training set to a CSV file
    training_df.to_csv(training_file, index=False)
    
    # Save the testing set to a CSV file
    testing_df.to_csv(testing_file, index=False)

# Assuming you have the training set and testing set as separate variables
#training_set = ...  #if it is necessary
#testing_set = ...   #if it is necessary

# Specify the file names for the training and testing datasets
training_file = "training_dataset.csv"
testing_file = "testing_dataset.csv"

# Save the training and testing datasets to CSV files
#save_dataset_to_csv(training_set, testing_set, training_file, testing_file)
save_dataset_to_csv(training_file, testing_file)
