"""
downloader/downloader.py

Parallel Image Downloader
"""

import os
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image


class ImageDownloader:

    def __init__(
        self,
        workers=16,
        retries=3,
        timeout=20,
        target_era="WWII"
    ):

        self.workers = workers
        self.retries = retries
        self.timeout = timeout

        # Force everything into this era folder.
        # Set to None to fall back to per-record year classification.
        self.target_era = target_era

        self.lock = threading.Lock()

        self.downloaded = 0
        self.failed = 0
        self.skipped = 0

    ############################################################

    def get_era(self, year):

        if year is None:
            return "Unknown"

        if year <= 1919:
            return "1900_1919"

        elif year <= 1939:
            return "1920_1939"

        elif year <= 1945:
            return "WWII"

        elif year <= 1959:
            return "1950s"

        elif year <= 1969:
            return "1960s"

        elif year <= 1979:
            return "1970s"

        else:
            return "Modern"

    ############################################################

    def filename(self, record):

        image_id = str(record.get("id", "unknown")).replace("/", "_")

        ext = ".jpg"

        url = str(record.get("image_url", "")).split("?")[0].lower()

        if ".png" in url:
            ext = ".png"

        return image_id + ext

    ############################################################

    def download_one(self, record):

        if self.target_era is not None:
            era = self.target_era
        else:
            era = self.get_era(record.get("year"))

        folder = Path("dataset") / era

        folder.mkdir(parents=True, exist_ok=True)

        file_path = folder / self.filename(record)

        if file_path.exists():

            with self.lock:
                self.skipped += 1

            return

        url = str(record.get("image_url", ""))

        # Reject missing URLs or non-image placeholders up front.
        if not url or url.split("?")[0].lower().endswith(".svg"):

            with self.lock:
                self.failed += 1

            return

        for attempt in range(self.retries):

            try:

                response = requests.get(
                    url,
                    timeout=self.timeout,
                    stream=True,
                    headers={
                        "User-Agent": "HistoricalDatasetBuilder/1.0"
                    }
                )

                response.raise_for_status()

                with open(file_path, "wb") as f:

                    for chunk in response.iter_content(8192):

                        if chunk:
                            f.write(chunk)

                # Verify image
                with Image.open(file_path) as img:
                    img.verify()

                with self.lock:

                    self.downloaded += 1

                    if self.downloaded % 100 == 0:

                        print(
                            f"Downloaded {self.downloaded}"
                        )

                return

            except Exception:

                if file_path.exists():
                    file_path.unlink()

        with self.lock:
            self.failed += 1

    ############################################################

    def download(self, metadata):

        print("\nStarting Parallel Download...\n")

        with ThreadPoolExecutor(
            max_workers=self.workers
        ) as executor:

            futures = [

                executor.submit(
                    self.download_one,
                    record
                )

                for record in metadata

            ]

            for future in as_completed(futures):

                try:
                    future.result()
                except Exception as e:
                    print(f"Worker error: {e}")

        print("\nDownload Finished")

        print("--------------------------------")

        print("Downloaded :", self.downloaded)

        print("Skipped    :", self.skipped)

        print("Failed     :", self.failed)

        print("--------------------------------")