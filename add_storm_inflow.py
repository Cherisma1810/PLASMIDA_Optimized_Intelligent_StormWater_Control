import os

inp_filename = "velachery_model.inp"

if not os.path.exists(inp_filename):
    print(f"Error: {inp_filename} not found.")
    exit()

print(f"Injecting Heavy Storm Event into {inp_filename}...")

with open(inp_filename, 'a') as f:
    # Define a heavy synthetic storm hydrograph
    f.write("\n[TIMESERIES]\n")
    f.write(";;Name           Date       Time       Value (CMS)\n")
    f.write("HeavyStorm       01/01/2026 00:00      0.0\n")
    f.write("HeavyStorm       01/01/2026 00:30      2.5\n") # Rapid escalation
    f.write("HeavyStorm       01/01/2026 01:00      6.0\n") # Peak intensity
    f.write("HeavyStorm       01/01/2026 01:30      3.5\n")
    f.write("HeavyStorm       01/01/2026 02:00      1.0\n")
    f.write("HeavyStorm       01/01/2026 03:00      0.0\n")
    
    # Inject the storm flow directly into the highest-elevation critical nodes
    f.write("\n[INFLOWS]\n")
    f.write(";;Node           Constituent      Time Series      Type     Mfactor  Sfactor  Baseline Pattern\n")
    inflow_nodes = ["A5", "C41", "B12", "C39", "A12"]
    for node in inflow_nodes:
        f.write(f"{node:<14} FLOW             HeavyStorm       FLOW     1.0      1.0      0.0\n")

print("Storm injected successfully! The system will now flood if pumps are not managed correctly.")