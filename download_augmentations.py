import os
import requests
import zipfile
import tarfile

def download_and_extract(url, extract_to):
    print(f"Downloading {url}...")
    filename = url.split('/')[-1]
    
    if not os.path.exists(filename):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
    print(f"Extracting {filename}...")
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif filename.endswith('.tgz') or filename.endswith('.tar.gz'):
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    print("Done.")

os.makedirs('augmentations/noise', exist_ok=True)
os.makedirs('augmentations/rir', exist_ok=True)

demand_urls = [
    "https://zenodo.org/records/1227121/files/PCAFETER_16k.zip",
    "https://zenodo.org/records/1227121/files/DKITCHEN_16k.zip",
    "https://zenodo.org/records/1227121/files/STRAFFIC_16k.zip",
    "https://zenodo.org/records/1227121/files/NPARK_16k.zip"
]
for url in demand_urls:
    download_and_extract(url, 'augmentations/noise')

openslr_url = "http://www.openslr.org/resources/28/rirs_noises.zip"
download_and_extract(openslr_url, 'augmentations/rir')

print("All acoustic augmentations downloaded successfully!")
