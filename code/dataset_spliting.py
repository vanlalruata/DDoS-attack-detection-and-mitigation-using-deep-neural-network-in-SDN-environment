import random

def preprocess_and_split_dataset(dataset, training_percentage):
    # Shuffle the dataset randomly
    random.shuffle(dataset)
    
    # Calculate the number of samples in the training set
    n_train = round(training_percentage * len(dataset))
    
    # Extract the first n_train samples as the training set
    training_set = dataset[:n_train]
    
    # Extract the remaining samples as the testing set
    testing_set = dataset[n_train:]
    
    # Perform any required data preprocessing steps on the training and testing sets
    
    # Return the preprocessed training and testing sets
    return training_set, testing_set

	# Assuming you have a dataset stored in a list called 'dataset'
	training_percentage = 0.8  # 80% of the dataset will be used for training
	
	# Call the preprocess_and_split_dataset function
	training_set, testing_set = preprocess_and_split_dataset(dataset, training_percentage)
	
	# Print the lengths of the training and testing sets
	print("Training set length:", len(training_set))
	print("Testing set length:", len(testing_set))



