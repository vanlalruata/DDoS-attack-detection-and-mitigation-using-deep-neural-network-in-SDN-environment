import time
import csv

def capture_attack_detection_time_vs_rate(num_attacks, attack_interval, output_file):
    attack_rates = []
    detection_times = []
    
    # Simulate the attacks and measure the detection time
    for i in range(num_attacks):
        start_time = time.time()
        
        # Perform the attack or detection operation
        # Replace the following line with your own attack or detection code
        time.sleep(attack_interval)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Calculate the attack rate and add it to the list
        attack_rate = 1 / elapsed_time
        attack_rates.append(attack_rate)
        
        # Add the detection time to the list
        detection_times.append(elapsed_time)
    
    # Write the data to the CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Attack Rate', 'Detection Time'])
        
        for i in range(num_attacks):
            writer.writerow([attack_rates[i], detection_times[i]])

# Set the number of attacks, attack interval, and output file path
num_attacks = 700  # Specify the number of attacks
attack_interval = 10  # Specify the time interval between attacks (in seconds)
output_file = 'attack_detection_data.csv'  # Specify the output file path and name

# Call the capture_attack_detection_time_vs_rate function
capture_attack_detection_time_vs_rate(num_attacks, attack_interval, output_file)
