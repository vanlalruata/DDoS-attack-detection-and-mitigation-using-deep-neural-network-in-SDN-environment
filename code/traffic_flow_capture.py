import pyshark
import csv

def capture_traffic_and_save_to_csv(interface, capture_duration, output_file):
    # Open a live capture session on the specified interface
    capture = pyshark.LiveCapture(interface=interface)
    
    # Start the capture
    capture.sniff(timeout=capture_duration)
    
    # Open the output file in write mode
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row
        writer.writerow(['Timestamp', 'Source IP', 'Destination IP', 'Protocol', 'Length'])
        
        # Iterate over captured packets and write data to the CSV file
        for packet in capture:
            timestamp = packet.sniff_time.strftime('%Y-%m-%d %H:%M:%S.%f')
            source_ip = packet.ip.src
            destination_ip = packet.ip.dst
            protocol = packet.transport_layer
            length = packet.length
            
            # Write the packet data to the CSV file
            writer.writerow([timestamp, source_ip, destination_ip, protocol, length])

# Set the capture interface, capture duration, and output file path
interface = 'eth0'  # Replace 'eth0' with the appropriate interface name on your system
capture_duration = 6000  # Specify the duration (in seconds) to capture traffic
output_file = 'traffic_data.csv'  # Specify the output file path and name

# Call the capture_traffic_and_save_to_csv function
capture_traffic_and_save_to_csv(interface, capture_duration, output_file)
