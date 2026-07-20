"""
Download YCB Objects for MuJoCo Simulation

Copyright 2015 Yale University - Grablab
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# Get the script directory and set output to ../models
script_dir = Path(__file__).parent
output_directory = script_dir.parent / "models" / "ycb"

# Download common household objects (10 most useful for manipulation)
objects_to_download = [
    "011_banana",
    "002_master_chef_can",      # Coffee can
    "003_cracker_box",           # Cheez-It box
    "005_tomato_soup_can",       # Soup can
    "006_mustard_bottle",        # Mustard bottle
    "024_bowl",                  # Bowl
    "025_mug",                   # Mug with handle
    "035_power_drill",           # Power drill
    "037_scissors",              # Scissors
    "040_large_marker",          # Sharpie marker
    "021_bleach_cleanser"        # Bleach bottle
]

# Download google 16k mesh for use in MuJoCo
files_to_download = ["google_16k"]

# Extract all files from the downloaded .tgz, and remove .tgz files
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"

if not output_directory.exists():
    output_directory.mkdir(parents=True)
    print(f"Created directory: {output_directory}")

def download_file(url, filename):
    """Download file with progress bar"""
    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.headers.get('Content-Length', 0))
            print(f"Downloading: {filename.name} ({file_size/1000000.0:.2f} MB)")
            
            downloaded = 0
            block_size = 65536
            
            with open(filename, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    if file_size > 0:
                        percent = downloaded * 100.0 / file_size
                        print(f"\r{downloaded/1000000.0:10.2f} MB  [{percent:3.2f}%]", end='')
            
            print()  # New line after download
            return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False

def tgz_url(object_name, file_type):
    """Generate URL for the file to download"""
    if file_type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return f"{base_url}berkeley/{object_name}/{object_name}_{file_type}.tgz"
    elif file_type in ["berkeley_processed"]:
        return f"{base_url}berkeley/{object_name}/{object_name}_berkeley_meshes.tgz"
    else:
        return f"{base_url}google/{object_name}_{file_type}.tgz"

def extract_tgz(filename, target_dir):
    """Extract tar.gz file and remove archive"""
    print(f"Extracting {filename.name}...")
    try:
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=target_dir)
        os.remove(filename)
        print(f"Extracted and removed {filename.name}")
        return True
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        return False

def check_url(url):
    """Check if URL exists"""
    try:
        request = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(request) as response:
            return response.status == 200
    except Exception:
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("YCB Object Downloader")
    print("=" * 60)
    
    for obj in objects_to_download:
        # Check if object already exists
        obj_dir = output_directory / obj / "google_16k"
        if obj_dir.exists() and (obj_dir / "textured.obj").exists():
            print(f"\n✓ {obj} already downloaded, skipping...")
            continue
        
        print(f"\nProcessing object: {obj}")
        for file_type in files_to_download:
            url = tgz_url(obj, file_type)
            
            print(f"Checking URL: {url}")
            if not check_url(url):
                print(f"URL not available, skipping: {url}")
                continue
            
            filename = output_directory / f"{obj}_{file_type}.tgz"
            
            if download_file(url, filename):
                if extract:
                    extract_tgz(filename, output_directory)
    
    print("\n" + "=" * 60)
    print(f"Download complete! Files saved to: {output_directory}")
    print("=" * 60)
    
    # List downloaded objects
    if output_directory.exists():
        print(f"\nDownloaded objects:")
        for obj_dir in sorted(output_directory.iterdir()):
            if obj_dir.is_dir():
                print(f"  ✓ {obj_dir.name}")
