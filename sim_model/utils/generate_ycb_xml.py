"""
Generate MuJoCo XML snippet for YCB objects
Run this after downloading YCB objects to generate the XML to paste into GIRAF.xml
"""

import random
from pathlib import Path

# YCB objects with their names
ycb_objects = {
    "011_banana": {"name": "Banana", "density": 600},
    "002_master_chef_can": {"name": "Coffee Can", "density": 200},
    "003_cracker_box": {"name": "Cracker Box", "density": 100},
    "005_tomato_soup_can": {"name": "Soup Can", "density": 300},
    "006_mustard_bottle": {"name": "Mustard Bottle", "density": 400},
    "024_bowl": {"name": "Bowl", "density": 500},
    "025_mug": {"name": "Mug", "density": 600},
    "035_power_drill": {"name": "Power Drill", "density": 800},
    "037_scissors": {"name": "Scissors", "density": 700},
    "040_large_marker": {"name": "Marker", "density": 200},
    "021_bleach_cleanser": {"name": "Bleach Bottle", "density": 400}
}

# Position bounds
x_range = (0.5, 1.0)
y_range = (-0.25, 0.25)
z_range = (0.1, 0.2)

# Check which objects exist
script_dir = Path(__file__).parent
ycb_dir = script_dir.parent / "models" / "ycb"

available_objects = []
for obj_id in ycb_objects.keys():
    obj_path = ycb_dir / obj_id / "google_16k"
    if obj_path.exists() and (obj_path / "textured.obj").exists():
        available_objects.append(obj_id)

if not available_objects:
    print("No YCB objects found! Run download_ycb_banana.py first.")
    exit(1)

print("=" * 80)
print("YCB Objects XML Generator")
print("=" * 80)
print(f"\nFound {len(available_objects)} objects")
print(f"Placing objects in: x={x_range}, y={y_range}, z={z_range}")
print("\n" + "=" * 80)

# Generate asset section
print("\n<!-- PASTE THIS IN <asset> SECTION -->\n")
print("    <!-- YCB Object meshes and textures -->")
for obj_id in available_objects:
    obj_name = ycb_objects[obj_id]["name"]
    print(f"    <mesh name=\"{obj_id}_mesh\" file=\"ycb/{obj_id}/google_16k/textured.obj\"/>")
    print(f"    <texture name=\"{obj_id}_tex\" type=\"2d\" file=\"ycb/{obj_id}/google_16k/texture_map.png\"/>")
    print(f"    <material name=\"{obj_id}_mat\" texture=\"{obj_id}_tex\" specular=\"0.3\" shininess=\"0.5\"/>")

# Generate worldbody section
print("\n\n<!-- PASTE THIS IN <worldbody> SECTION -->\n")
print("    <!-- YCB Objects -->")

random.seed(42)  # Reproducible positions
for obj_id in available_objects:
    obj_info = ycb_objects[obj_id]
    
    # Random position within bounds
    x = random.uniform(*x_range)
    y = random.uniform(*y_range)
    z = random.uniform(*z_range)
    
    # Random small rotation for variety
    angle = random.uniform(0, 6.28)
    
    print(f"    <body name=\"{obj_id}\" pos=\"{x:.3f} {y:.3f} {z:.3f}\" euler=\"0 0 {angle:.2f}\">")
    print(f"      <freejoint/>")
    print(f"      <geom name=\"{obj_id}_geom\" type=\"mesh\" mesh=\"{obj_id}_mesh\" material=\"{obj_id}_mat\" ")
    print(f"            density=\"{obj_info['density']}\" friction=\"2.0 0.1 0.01\" solref=\"0.008 1\" condim=\"6\"/>")
    print(f"    </body>")

print("\n" + "=" * 80)
print("Copy the sections above into your GIRAF.xml file")
print("=" * 80)
