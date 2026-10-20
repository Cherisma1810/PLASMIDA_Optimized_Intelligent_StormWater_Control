import json
import geopandas as gpd

# 1. Load Nodes
with open('ward_175.geojson', 'r') as file:
    node_data = json.load(file)

nodes = []
for f in node_data.get('features', []):
    props = f.get('properties', {})
    m_id = props.get('Manhole_ID')
    t_elev = props.get('Drain_Top_m')
    i_elev = props.get('Invert_m')
    depth = props.get('Depth_m')
    
    if m_id and t_elev and i_elev and str(t_elev).strip() != "-" and str(i_elev).strip() != "-":
        clean_id = str(m_id).replace(" ", "_").replace("(", "").replace(")", "")
        
        raw_depth = float(depth) if str(depth).strip() != "-" else 2.0
        # Force a minimum depth of 2.0m so nodes safely contain the 1.0m diameter pipes
        safe_depth = max(raw_depth, 2.0) 
        
        nodes.append({
            'id': clean_id,
            'invert': float(i_elev),
            'depth': safe_depth
        })

# Sort nodes from highest elevation to lowest (Gravity flow)
nodes = sorted(nodes, key=lambda x: x['invert'], reverse=True)

# 2. Load Conduits (Lines)
conduits_gdf = gpd.read_file('ward_175_lines.geojson').to_crs(epsg=32644)
raw_lengths = [row.geometry.length for idx, row in conduits_gdf.iterrows() if row.geometry.length > 0]

# Clamp lengths between 10m and 2000m to prevent the 102km OSM outlier from breaking the physics engine
lengths = [min(max(l, 10.0), 2000.0) for l in raw_lengths]

# 3. Write EPA SWMM .inp File
inp_filename = "velachery_model.inp"
print(f"Generating SWMM Model: {inp_filename}...")

with open(inp_filename, 'w') as f_out:
    f_out.write("[TITLE]\nVelachery Ward 175 Synthetic Drainage Network\n\n")
    
    # ALL Time parameters MUST go under OPTIONS in SWMM 5
    f_out.write("[OPTIONS]\n")
    f_out.write("FLOW_UNITS           CMS\n")
    f_out.write("INFILTRATION         HORTON\n")
    f_out.write("FLOW_ROUTING         DYNWAVE\n")
    f_out.write("START_DATE           01/01/2026\n")
    f_out.write("START_TIME           00:00:00\n")
    f_out.write("REPORT_START_DATE    01/01/2026\n")
    f_out.write("REPORT_START_TIME    00:00:00\n")
    f_out.write("END_DATE             01/02/2026\n")
    f_out.write("END_TIME             00:00:00\n")
    f_out.write("REPORT_STEP          00:05:00\n")
    f_out.write("WET_STEP             00:05:00\n")
    f_out.write("DRY_STEP             00:05:00\n")
    f_out.write("ROUTING_STEP         00:01:00\n\n")
    
    # Force SWMM to write error logs
    f_out.write("[REPORT]\n")
    f_out.write("INPUT      YES\n")
    f_out.write("CONTROLS   YES\n")
    f_out.write("NODES      ALL\n")
    f_out.write("LINKS      ALL\n\n")
    
    f_out.write("[JUNCTIONS]\n;;Name           Elevation  MaxDepth\n")
    for node in nodes[:-1]:
        f_out.write(f"{node['id']:<14} {node['invert']:<10.3f} {node['depth']:<10.3f}\n")
    f_out.write("\n")
    
    outfall = nodes[-1]
    f_out.write("[OUTFALLS]\n;;Name           Elevation  Type\n")
    f_out.write(f"{outfall['id']:<14} {outfall['invert']:<10.3f} FREE\n\n")
    
    f_out.write("[ORIFICES]\n;;Name           From Node        To Node          Type         Offset     Qcoeff     Gated\n")
    for i in range(7):
        from_node = nodes[i]['id']
        to_node = nodes[i+1]['id']
        link_id = f"Gate_{i+1}"
        f_out.write(f"{link_id:<14} {from_node:<16} {to_node:<16} SIDE         0          0.65       YES\n")
    f_out.write("\n")

    f_out.write("[CONDUITS]\n;;Name           From Node        To Node          Length     Roughness  InOffset   OutOffset\n")
    for i in range(7, len(nodes)-1):
        from_node = nodes[i]['id']
        to_node = nodes[i+1]['id']
        length = lengths[i % len(lengths)] 
        link_id = f"Pipe_{i+1}"
        f_out.write(f"{link_id:<14} {from_node:<16} {to_node:<16} {length:<10.2f} 0.015      0          0\n")
    f_out.write("\n")
    
    f_out.write("[XSECTIONS]\n;;Link           Shape        Geom1      Geom2      Geom3      Geom4      Barrels\n")
    for i in range(7):
        link_id = f"Gate_{i+1}"
        f_out.write(f"{link_id:<14} CIRCULAR     1.0        0          0          0          1\n") 
    for i in range(7, len(nodes)-1):
        link_id = f"Pipe_{i+1}"
        f_out.write(f"{link_id:<14} CIRCULAR     1.0        0          0          0          1\n")

print(f"Successfully generated {inp_filename} with {len(nodes)} nodes, 7 Gates, and {len(nodes)-1 - 7} Pipes.")