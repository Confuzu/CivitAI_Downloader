
import requests
import os
from tqdm import tqdm 
import argparse
from concurrent.futures import ThreadPoolExecutor

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download files from direct URLs provided in a file.")
parser.add_argument("--url_file", type=str, required=True, help="Path to the file containing direct download URLs.")
parser.add_argument("--max_threads", type=int, default=5, help="Maximum number of concurrent download threads.")
args = parser.parse_args()

# Create a session object for making multiple requests
session = requests.Session()

def file_already_exists(filename):
    return (os.path.exists(os.path.join("embeddings", filename)) or
            os.path.exists(os.path.join("loras", filename)) or
            os.path.exists(os.path.join("models", filename)))

def download_file_from_url(file_url_pair):
    filename, url = file_url_pair
    if file_already_exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return filename
    if not (filename.endswith(".pt") or filename.endswith(".safetensors")):
        print(f"Skipping {filename} as it does not have a valid extension.")
        return None
    try:
        response = session.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, leave=False, desc=filename)
        output_path = os.path.join(os.getcwd(), filename)
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))
        if output_path.endswith(".pt"):
            move_file_to_folder(output_path, "embeddings")
        elif os.path.getsize(output_path) < 2 * (1024 ** 3):  # Less than 2 GB
            move_file_to_folder(output_path, "loras")
        else:
            move_file_to_folder(output_path, "models")
        return filename
    except Exception as e:
        print(f"Error downloading {url}. Error: {e}")
        return None

def read_urls_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    urls = [line.split(" - ")[1].strip() for line in lines if " - " in line]
    filenames = [line.split(" - ")[0].strip() for line in lines if " - " in line]
    return list(zip(filenames, urls))

def move_file_to_folder(file_path, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    new_path = os.path.join(folder_name, os.path.basename(file_path))
    os.rename(file_path, new_path)

file_url_pairs = read_urls_from_file(args.url_file)

# Retry mechanism for failed downloads
max_attempts = 3
for attempt in range(max_attempts):
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        results = list(executor.map(download_file_from_url, file_url_pairs))
    failed_downloads = [file_url_pairs[i] for i, result in enumerate(results) if result is None]
    if not failed_downloads:
        print("All files downloaded successfully!")
        break
    else:
        print(f"{len(failed_downloads)} files failed to download. Retrying...")
        file_url_pairs = failed_downloads
else:
    print("Some files could not be downloaded after maximum attempts.")
    print("The following files could not be downloaded:")
    for filename, _ in failed_downloads:
        print(filename)
