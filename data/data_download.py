import requests
from bs4 import BeautifulSoup
import os
import time
import concurrent.futures
from urllib.parse import urljoin

URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"

OUTPUT_DIR = "daic_woz_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WORKERS = 10

total_files = 0
downloaded_count = 0
failed_count = 0


def download_file(url, filename):
    global downloaded_count, failed_count

    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        print(f"✓ File exists: {filename}")
        downloaded_count += 1
        return True

    print(f"↓ Downloading: {filename}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        with requests.get(url, stream=True, headers=headers, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            start_time = time.time()

            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded += len(chunk)
                        f.write(chunk)
                        if total_size > 0:
                            progress = int(50 * downloaded / total_size)
                            speed = downloaded / (time.time() - start_time) / 1024
                            print(f"\r[{filename[:20]:<20}] [{'=' * progress}{' ' * (50 - progress)}] "
                                  f"{downloaded / 1024 / 1024:.1f}MB/{total_size / 1024 / 1024:.1f}MB "
                                  f"@ {speed:.1f}KB/s", end='')

        downloaded_count += 1
        print(f"\n✓ Download complete: {filename} (Size: {total_size / 1024 / 1024:.1f}MB)")
        return True

    except Exception as e:
        failed_count += 1
        print(f"\n✗ Download failed: {filename} - {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def get_download_links():
    print(f"Parsing website: {URL}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(URL, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Unable to access website: {str(e)}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    download_links = []
    for link in soup.find_all('a'):
        href = link.get('href', '')
        text = link.text.strip()

        if (href.endswith(('.zip', '.tar', '.gz', '.7z', '.rar', '.csv', '.txt', '.json', '.xlsx')) or
                "download" in href.lower() or
                "dataset" in text.lower() or
                "data" in text.lower()):

            full_url = urljoin(URL, href)
            filename = os.path.basename(full_url) or f"dataset_file_{len(download_links) + 1}"

            if not os.path.splitext(filename)[1]:
                if "zip" in text.lower():
                    filename += ".zip"
                elif "tar" in text.lower():
                    filename += ".tar"
                elif "csv" in text.lower():
                    filename += ".csv"
                else:
                    filename += ".dat"

            download_links.append((full_url, filename))

    return download_links


def main():
    global total_files

    download_links = get_download_links()
    if not download_links:
        print("No download links found, please check website structure")
        return

    total_files = len(download_links)
    print(f"Found {total_files} files to download")
    print("=" * 70)

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(download_file, url, filename): (url, filename)
                          for url, filename in download_links}

        for future in concurrent.futures.as_completed(future_to_file):
            url, filename = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                print(f"File download error: {filename} - {str(e)}")

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Download complete! Statistics:")
    print(f"Total files: {total_files}")
    print(f"Successful downloads: {downloaded_count}")
    print(f"Failed downloads: {failed_count}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average speed: {downloaded_count / total_time:.2f} files/second" if downloaded_count > 0 else "")
    print("Files saved in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()