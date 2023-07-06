import pyshark
import csv

def capture_incoming_packets(scenario, output_file):
    # Set the capture filter based on the scenario
    if scenario == "Attack without Mitigation":
        capture_filter = "your_capture_filter_here"
    elif scenario == "Attack with Mitigation":
        capture_filter = "your_capture_filter_here"
    elif scenario == "Attack Free":
        capture_filter = "your_capture_filter_here"
    else:
        raise ValueError("Invalid scenario")
    
    # Open a live capture session
    capture = pyshark.LiveCapture(display_filter=capture_filter)
    
    # Open the output file in write mode
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row
        writer.writerow(['Source IP', 'Destination IP', 'Protocol'])
        
        # Capture packets and write data to the CSV file
        for packet in capture.sniff_continuously():
            source_ip = packet.ip.src
            destination_ip = packet.ip.dst
            protocol = packet.transport_layer
            writer.writerow([source_ip, destination_ip, protocol])

# Set the scenario and output file path
scenario = "Attack without Mitigation"  # Specify the scenario: "Attack without Mitigation", "Attack with Mitigation", or "Attack Free"
output_file = f"{scenario.replace(' ', '_')}_incoming_packets.csv"  # Specify the output file path and name

# Call the capture_incoming_packets function
capture_incoming_packets(scenario, output_file)
