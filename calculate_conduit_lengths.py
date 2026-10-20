import geopandas as gpd

# 1. Load the newly found conduit line data
line_file = 'ward_175_lines.geojson'

try:
    # Read the spatial file
    conduits = gpd.read_file(line_file)
    
    # 2. Project to a metric Coordinate Reference System (CRS) to calculate accurate lengths in meters.
    # EPSG:32644 is the UTM zone for Chennai, India.
    if conduits.crs is None or conduits.crs.to_string() != 'EPSG:32644':
        print("Projecting spatial data to EPSG:32644 (Meters)...")
        conduits = conduits.to_crs(epsg=32644)
    
    print(f"{'Conduit_ID':<15} | {'Length (Meters)':<15}")
    print("-" * 35)

    # 3. Calculate and print the length of each conduit
    for index, row in conduits.iterrows():
        # Assuming the file has an ID column, modify 'id' if the column is named differently (e.g., 'Drain_ID')
        conduit_id = row.get('id', f"Link_{index}") 
        length_m = row.geometry.length
        
        print(f"{conduit_id:<15} | {length_m:.2f}")

except FileNotFoundError:
    print(f"Error: {line_file} not found. Please download the line data from OpenCity first.")