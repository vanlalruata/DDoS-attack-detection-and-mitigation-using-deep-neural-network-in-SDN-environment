import psutil
import csv
import time

def capture_cpu_usage_and_save_to_csv(capture_duration, output_file):
    # Open the output file in write mode
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row
        writer.writerow(['Timestamp', 'CPU Usage'])
        
        # Capture CPU usage for the specified duration
        start_time = time.time()
        end_time = start_time + capture_duration
        
        while time.time() < end_time:
            # Retrieve the current CPU usage percentage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Get the current timestamp
            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Write the timestamp and CPU usage to the CSV file
            writer.writerow([current_time, cpu_usage])
            
            # Sleep for a short duration (e.g., 1 second) before capturing the next CPU usage
            time.sleep(1)

# Set the capture duration and output file path
capture_duration = 6000  # Specify the duration (in seconds) to capture CPU usage
output_file = 'cpu_usage_data.csv'  # Specify the output file path and name

# Call the capture_cpu_usage_and_save_to_csv function
capture_cpu_usage_and_save_to_csv(capture_duration, output_file)
