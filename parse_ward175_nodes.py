import json
import os

# Load the Ward 175 GeoJSON file
file_path = 'ward_175.geojson'

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found in the current directory.")
    exit()

with open(file_path, 'r') as file:
    data = json.load(file)

print(f"{'Manhole_ID':<15} | {'Top Elev (m)':<15} | {'Invert Elev (m)':<15} | {'Max Depth (m)':<15}")
print("-" * 65)

# Iterate through the features to extract SWMM parameters
for feature in data.get('features', []):
    props = feature.get('properties', {})
    
    manhole_id = props.get('Manhole_ID')
    top_elev = props.get('Drain_Top_m')
    invert_elev = props.get('Invert_m')
    max_depth = props.get('Depth_m')
    
    # Filter out empty or null records
    if manhole_id and top_elev and invert_elev and str(top_elev).strip() != "-":
        print(f"{manhole_id:<15} | {top_elev:<15} | {invert_elev:<15} | {max_depth:<15}")