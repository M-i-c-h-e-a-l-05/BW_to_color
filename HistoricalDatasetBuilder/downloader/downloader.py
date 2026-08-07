"""
downloader/downloader.py

Parallel Image Downloader
WWII Dataset Collection Version

Features:
- Parallel downloading
- Automatic retries
- Image verification
- Resume support
- Forced destination folder
- Skips existing files

Author: Micheal Leveiro
"""

import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image


class ImageDownloader:

    def __init__(
        self,
        workers=16,
        retries=5,
        timeout=30,
        target_era=None
    ):

        self.workers = workers
        self.retries = retries
        self.timeout = timeout

        # If target_era is provided, ALL downloaded
        # images will be placed there.
        #
        # Example:
        # target_era="WWII"

        self.target_era = target_era

        self.lock = threading.Lock()

        self.downloaded = 0
        self.failed = 0
        self.skipped = 0

    # ============================================================
    # NORMAL ERA CLASSIFICATION
    # ============================================================

    def get_era(self, year):

        if year is None:
            return "Unknown"

        if year <= 1919:
            return "1900"

        elif year <= 1939:
            return "1920"

        elif year <= 1945:
            return "WWII"

        elif year <= 1959:
            return "1950"

        elif year <= 1969:
            return "1960"

        elif year <= 1979:
            return "1970"

        else:
            return "Modern"

    # ============================================================
    # FILENAME
    # ============================================================

    def filename(self, record):

        image_id = str(
            record.get("id", "unknown")
        ).replace("/", "_")

        image_id = image_id.replace("\\", "_")

        url = str(
            record.get("image_url", "")
        ).lower()

        ext = ".jpg"

        if ".png" in url:
            ext = ".png"

        elif ".webp" in url:
            ext = ".webp"

        elif ".tiff" in url:
            ext = ".tiff"

        elif ".tif" in url:
            ext = ".tif"

        return image_id + ext

    # ============================================================
    # DOWNLOAD ONE IMAGE
    # ============================================================

    def download_one(self, record):

        # --------------------------------------------------------
        # Determine destination
        # --------------------------------------------------------

        if self.target_era is not None:

            era = self.target_era

        else:

            era = self.get_era(
                record.get("year")
            )

        folder = (
            Path("dataset") /
            era
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------------
        # Filename
        # --------------------------------------------------------

        file_path = (
            folder /
            self.filename(record)
        )

        # --------------------------------------------------------
        # Skip existing image
        # --------------------------------------------------------

        if file_path.exists():

            with self.lock:
                self.skipped += 1

            return

        url = record.get("image_url")

        if not url:

            with self.lock:
                self.failed += 1

            return

        # ========================================================
        # RETRIES
        # ========================================================

        for attempt in range(
            1,
            self.retries + 1
        ):

            try:

                response = requests.get(

                    url,

                    timeout=self.timeout,

                    stream=True,

                    headers={
                        "User-Agent":
                        "HistoricalDatasetBuilder/1.0"
                    }

                )

                response.raise_for_status()

                # ------------------------------------------------
                # Write image
                # ------------------------------------------------

                with open(
                    file_path,
                    "wb"
                ) as f:

                    for chunk in response.iter_content(
                        chunk_size=8192
                    ):

                        if chunk:
                            f.write(chunk)

                # ------------------------------------------------
                # Verify image
                # ------------------------------------------------

                with Image.open(
                    file_path
                ) as image:

                    image.verify()

                # ------------------------------------------------
                # Success
                # ------------------------------------------------

                with self.lock:

                    self.downloaded += 1

                    count = self.downloaded

                    if count % 100 == 0:

                        print(
                            f"Downloaded: {count}"
                        )

                return

            except Exception as e:

                # Remove incomplete/corrupt file

                if file_path.exists():

                    try:
                        file_path.unlink()
                    except Exception:
                        pass

                if attempt < self.retries:

                    print(
                        f"Retry "
                        f"{attempt}/{self.retries}: "
                        f"{url}"
                    )

        # ========================================================
        # FAILED
        # ========================================================

        with self.lock:

            self.failed += 1

    # ============================================================
    # DOWNLOAD ALL
    # ============================================================

    def download(self, metadata):

        print()
        print("=" * 60)
        print("STARTING PARALLEL DOWNLOAD")
        print("=" * 60)

        print(
            f"Records       : {len(metadata)}"
        )

        print(
            f"Workers       : {self.workers}"
        )

        print(
            f"Retries       : {self.retries}"
        )

        if self.target_era:

            print(
                f"Destination   : dataset/"
                f"{self.target_era}"
            )

        else:

            print(
                "Destination   : "
                "year-based classification"
            )

        print("=" * 60)
        print()

        # Reset counters

        self.downloaded = 0
        self.failed = 0
        self.skipped = 0

        # ========================================================
        # THREAD POOL
        # ========================================================

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

            for future in as_completed(
                futures
            ):

                try:

                    future.result()

                except Exception as e:

                    print(
                        f"Worker error: {e}"
                    )

        # ========================================================
        # SUMMARY
        # ========================================================

        print()
        print("=" * 60)
        print("DOWNLOAD FINISHED")
        print("=" * 60)

        print(
            f"Downloaded : {self.downloaded}"
        )

        print(
            f"Skipped    : {self.skipped}"
        )

        print(
            f"Failed     : {self.failed}"
        )

        print("=" * 60)
