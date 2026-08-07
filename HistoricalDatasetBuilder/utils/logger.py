"""
utils/logger.py

Research Logger

Author: Micheal Leveiro
"""

import logging
from pathlib import Path


class DatasetLogger:

    def __init__(

        self,

        log_folder="logs",

        log_name="dataset.log"

    ):

        Path(log_folder).mkdir(

            parents=True,

            exist_ok=True

        )

        self.logger = logging.getLogger(

            "HistoricalDataset"

        )

        self.logger.setLevel(

            logging.INFO

        )

        if self.logger.handlers:
            return

        formatter = logging.Formatter(

            "%(asctime)s | %(levelname)s | %(message)s"

        )

        file_handler = logging.FileHandler(

            Path(log_folder) / log_name,

            encoding="utf-8"

        )

        file_handler.setFormatter(

            formatter

        )

        console_handler = logging.StreamHandler()

        console_handler.setFormatter(

            formatter

        )

        self.logger.addHandler(

            file_handler

        )

        self.logger.addHandler(

            console_handler

        )

    ########################################################

    def info(self, message):

        self.logger.info(message)

    ########################################################

    def warning(self, message):

        self.logger.warning(message)

    ########################################################

    def error(self, message):

        self.logger.error(message)

    ########################################################

    def download(self, url):

        self.logger.info(

            f"DOWNLOADED | {url}"

        )

    ########################################################

    def failed(self, url, reason):

        self.logger.error(

            f"FAILED | {url} | {reason}"

        )

    ########################################################

    def duplicate(self, image):

        self.logger.info(

            f"DUPLICATE REMOVED | {image}"

        )

    ########################################################

    def quality_removed(self, image):

        self.logger.info(

            f"QUALITY REMOVED | {image}"

        )

    ########################################################

    def api(self, source, message):

        self.logger.info(

            f"{source} | {message}"

        )

    ########################################################

    def statistics(

        self,

        downloaded,

        failed,

        duplicates,

        removed

    ):

        self.logger.info("")

        self.logger.info(

            "============== SUMMARY =============="

        )

        self.logger.info(

            f"Downloaded : {downloaded}"

        )

        self.logger.info(

            f"Failed : {failed}"

        )

        self.logger.info(

            f"Duplicates : {duplicates}"

        )

        self.logger.info(

            f"Quality Removed : {removed}"

        )

        self.logger.info(

            "====================================="
        )