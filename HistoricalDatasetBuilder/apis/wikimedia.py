"""
apis/wikimedia.py

Improved Wikimedia Commons API Client
Research Version

Author: Micheal Leveiro
"""

import time
import requests

from utils.year_parser import YearParser


class WikimediaAPI:

    BASE_URL = "https://commons.wikimedia.org/w/api.php"

    SEARCH_TERMS = [

        # # General
        # "historical photograph",
        # "historic photograph",
        # "old photograph",

        # # People
        # "portrait",
        # "family",
        # "children",
        # "women",
        # "men",
        # "workers",
        # "soldiers",
        # "students",

        # # Cities
        # "street",
        # "street scene",
        # "city",
        # "town",
        # "village",
        # "market",
        # "harbor",

        # # Transport
        # "railway",
        # "train",
        # "tram",
        # "automobile",
        # "bus",
        # "ship",
        # "airplane",

        # # Buildings
        # "school",
        # "church",
        # "factory",
        # "bridge",
        # "library",
        # "museum",

        # # Historical Events
        # "World War I",
        "World War II"
        # "Great Depression",

        # # Lifestyle
        # "festival",
        # "farm",
        # "industry",
        # "construction",
        # "agriculture",

        # # Misc
        # "vintage",
        # "archive",
        # "newspaper photograph",
        # "black and white photograph"
    ]

    ##############################################################

    def __init__(

        self,

        max_records=750,

        delay=0.5,

        retries=5

    ):

        self.max_records = max_records

        self.delay = delay

        self.retries = retries

        self.year_parser = YearParser()

        self.session = requests.Session()

        self.session.headers.update({

            "User-Agent":
            "HistoricalDatasetBuilder/1.0 (Research Dataset)"

        })

    ##############################################################

    def request_with_retry(self, params):

        wait = 1

        for attempt in range(self.retries):

            try:

                response = self.session.get(

                    self.BASE_URL,

                    params=params,

                    timeout=30

                )

                response.raise_for_status()

                return response.json()

            except Exception as e:

                print(

                    f"Retry {attempt+1}/{self.retries} : {e}"

                )

                time.sleep(wait)

                wait *= 2

        return None

    ##############################################################

    def collect(self):

        records = []

        seen = set()

        for term in self.SEARCH_TERMS:

            print(f"\nSearching: {term}")

            new_records = self.search_term(

                term,

                seen

            )

            records.extend(new_records)

            print(

                f"Total records: {len(records)}"

            )

            if len(records) >= self.max_records:

                break

        return records[:self.max_records]

    ##############################################################

    def search_term(

        self,

        term,

        seen

    ):

        collected = []

        params = {

            "action": "query",

            "format": "json",

            "generator": "search",

            "gsrsearch": term,

            "gsrnamespace": 6,

            "gsrlimit": 50,

            "prop": "imageinfo|info",

            "iiprop": "url|extmetadata"

        }

        while True:

            data = self.request_with_retry(params)

            if data is None:

                break

            pages = data.get(

                "query",

                {}

            ).get(

                "pages",

                {}

            )

            if len(pages) == 0:

                break

            for page in pages.values():

                record = self.parse_item(page)

                if record is None:

                    continue

                if record["id"] in seen:

                    continue

                seen.add(

                    record["id"]

                )

                collected.append(record)

            print(

                f"{term}: {len(collected)} images"

            )

            if "continue" not in data:

                break

            params.update(

                data["continue"]

            )

            time.sleep(

                self.delay

            )

        return collected


        ##############################################################
    # Helper Functions
    ##############################################################

    def clean_html(self, text):

        """
        Remove HTML tags from Wikimedia metadata.
        """

        if not text:
            return ""

        import re

        text = re.sub(r"<[^>]+>", "", str(text))

        text = (
            text.replace("&quot;", "\"")
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&#39;", "'")
        )

        return text.strip()

    ##############################################################

    def get_year(self, metadata, title):

        """
        Extract year using the shared YearParser.
        """

        fields = [

            "DateTimeOriginal",

            "DateTime",

            "Date",

            "ObjectName",

            "ImageDescription"

        ]

        for field in fields:

            if field in metadata:

                value = metadata[field].get(

                    "value",

                    ""

                )

                value = self.clean_html(value)

                year = self.year_parser.parse(value)

                if year is not None:

                    return year

        return self.year_parser.parse(title)

    ##############################################################

    def extract_license(self, metadata):

        """
        Extract license from several possible fields.
        """

        fields = [

            "LicenseShortName",

            "License",

            "UsageTerms"

        ]

        for field in fields:

            if field in metadata:

                value = metadata[field].get(

                    "value",

                    ""

                )

                value = self.clean_html(value)

                if value != "":

                    return value

        return "Unknown"

    ##############################################################

    def extract_country(self, metadata):

        """
        Country is rarely present on Wikimedia,
        but check several fields.
        """

        fields = [

            "Country",

            "Location",

            "Place"

        ]

        for field in fields:

            if field in metadata:

                value = metadata[field].get(

                    "value",

                    ""

                )

                value = self.clean_html(value)

                if value != "":

                    return value

        return ""

    ##############################################################

    def is_photograph(self, metadata, title):

        """
        Reject obvious non-photographs.
        """

        text = title.lower()

        if "ImageDescription" in metadata:

            text += " "

            text += self.clean_html(

                metadata["ImageDescription"].get(

                    "value",

                    ""

                )

            ).lower()

        blocked = [

            "painting",

            "drawing",

            "map",

            "poster",

            "advertisement",

            "coin",

            "stamp",

            "manuscript",

            "book",

            "newspaper",

            "illustration",

            "engraving",

            "woodcut",

            "lithograph",

            "sculpture"

        ]

        for word in blocked:

            if word in text:

                return False

        return True

    ##############################################################

    def normalize_record(self, record):

        """
        Standardize output format.
        """

        return {

            "id": str(record["id"]),

            "title": record["title"],

            "year": record["year"],

            "image_url": record["image_url"],

            "source": "Wikimedia Commons",

            "license": record["license"],

            "country": record["country"],

            "type": "IMAGE"

        }

    ##############################################################

    def parse_item(self, page):

        """
        Convert Wikimedia page into a standard record.
        """

        imageinfo = page.get("imageinfo")

        if not imageinfo:

            return None

        info = imageinfo[0]

        image_url = info.get("url")

        if image_url is None:

            return None

        metadata = info.get(

            "extmetadata",

            {}

        )

        title = page.get(

            "title",

            ""

        )

        if not self.is_photograph(

            metadata,

            title

        ):

            return None

        record = {

            "id": page.get(

                "pageid"

            ),

            "title": title,

            "year": self.get_year(

                metadata,

                title

            ),

            "image_url": image_url,

            "license": self.extract_license(

                metadata

            ),

            "country": self.extract_country(

                metadata

            )

        }

        return self.normalize_record(record)