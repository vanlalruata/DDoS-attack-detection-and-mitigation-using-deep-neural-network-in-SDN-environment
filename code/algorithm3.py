import numpy as np

def detect_ddos_attack(network_traffic_data, model_weights, model_biases, threshold):
    detected_attacks = []
    
    # Load the DNN model weights and biases
    weights = model_weights
    biases = model_biases
    
    for i in range(len(network_traffic_data)):
        x = network_traffic_data[i]
        
        # Forward propagation
        activation = x
        for j in range(len(weights)):
            weighted_sum = np.dot(weights[j], activation) + biases[j]
            activation = sigmoid(weighted_sum)  # Apply the activation function of your choice
        
        # Obtain the output prediction
        y_hat = activation
        
        # Check if the prediction exceeds the threshold
        if y_hat > threshold:
            detected_attacks.append(x)
    
    return detected_attacks
