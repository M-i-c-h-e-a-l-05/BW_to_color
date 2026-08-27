"""
downloader/metadata.py

Metadata Manager
Author: Micheal Leveiro
"""

import csv
import os


class MetadataManager:

    def __init__(self):

        self.csv_file = "metadata/metadata.csv"

        os.makedirs("metadata", exist_ok=True)

    #########################################################

    def normalize(self, record):

        """
        Ensure every record has the same fields.
        """

        return {

            "id": str(record.get("id", "")),

            "title": record.get("title", ""),

            "year": record.get("year", None),

            "image_url": record.get("image_url", ""),

            "source": record.get("source", ""),

            "license": record.get("license", ""),

            "country": record.get("country", ""),

            "type": record.get("type", "")

        }

    #########################################################

    def remove_duplicate_metadata(self, metadata):

        unique = []

        seen_ids = set()

        seen_urls = set()

        for record in metadata:

            record = self.normalize(record)

            image_id = record["id"]

            image_url = record["image_url"]

            if image_id in seen_ids:
                continue

            if image_url in seen_urls:
                continue

            seen_ids.add(image_id)

            seen_urls.add(image_url)

            unique.append(record)

        return unique

    #########################################################

    def save_csv(self, metadata):

        metadata = self.remove_duplicate_metadata(metadata)

        fields = [

            "id",

            "title",

            "year",

            "image_url",

            "source",

            "license",

            "country",

            "type"

        ]

        with open(
            self.csv_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()

            writer.writerows(metadata)

        print(f"Saved {len(metadata)} records")

    #########################################################

    def load_csv(self):

        if not os.path.exists(self.csv_file):

            return []

        metadata = []

        with open(
            self.csv_file,
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                if row["year"] != "":

                    row["year"] = int(row["year"])

                else:

                    row["year"] = None

                metadata.append(row)

        print(f"Loaded {len(metadata)} records")

        return metadata

    #########################################################

    def append(self, record):

        """
        Append a single record.
        Useful for long downloads.
        """

        exists = os.path.exists(self.csv_file)

        fields = [

            "id",

            "title",

            "year",

            "image_url",

            "source",

            "license",

            "country",

            "type"

        ]

        with open(
            self.csv_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            if not exists:

                writer.writeheader()

            writer.writerow(record)

    #########################################################

    def statistics(self, metadata):

        print("\nMetadata Statistics")

        print("-" * 40)

        print("Total :", len(metadata))

        by_source = {}

        for record in metadata:

            source = record["source"]

            by_source[source] = by_source.get(source, 0) + 1

        print()

        for source, count in by_source.items():

            print(f"{source:<25}{count}")

        print("-" * 40)