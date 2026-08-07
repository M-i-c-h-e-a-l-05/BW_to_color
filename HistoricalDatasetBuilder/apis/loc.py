"""
apis/loc.py

Improved Library of Congress API Client
Research Version

Author: Micheal Leveiro
"""

import time
import requests

from utils.year_parser import YearParser


class LOCAPI:

    BASE_URL = "https://www.loc.gov/photos/"

    SEARCH_TERMS = [

        # # General
        # "historic photograph",
        # "historical photograph",
        # "old photograph",

        # # People
        # "portrait",
        # "family",
        # "children",
        # "women",
        # "workers",
        # "soldiers",

        # # Cities
        # "street",
        # "street scene",
        # "city",
        # "town",
        # "village",
        # "market",

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

        # # Agriculture
        # "farm",
        # "agriculture",

        # # Industry
        # "construction",
        # "industry",

        # # Wars
        # "World War I",
        "World War II"

        # # Misc
        # "harbor",
        # "festival",
        # "black and white",
        # "archive"
    ]

    ###########################################################

    def __init__(

        self,

        rows=100,

        max_records=750,

        delay=0.5,

        retries=5

    ):

        self.rows = rows

        self.max_records = max_records

        self.delay = delay

        self.retries = retries

        self.year_parser = YearParser()

        self.session = requests.Session()

        self.session.headers.update({

            "User-Agent":
            "HistoricalDatasetBuilder/1.0 (Research Dataset)"

        })

    ###########################################################

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

                    f"Retry {attempt+1}/{self.retries}: {e}"

                )

                time.sleep(wait)

                wait *= 2

        return None

    ###########################################################

    def collect(self):

        records = []

        seen = set()

        for term in self.SEARCH_TERMS:

            print(f"\nSearching LOC: {term}")

            new_records = self.search_term(

                term,

                seen

            )

            records.extend(new_records)

            print(

                f"Total LOC records: {len(records)}"

            )

            if len(records) >= self.max_records:

                break

        return records[:self.max_records]

    ###########################################################

    def search_term(

        self,

        term,

        seen

    ):

        collected = []

        page = 1

        while True:

            params = {

                "fo": "json",

                "sp": page,

                "c": self.rows,

                "q": term

            }

            data = self.request_with_retry(params)

            if data is None:

                break

            results = data.get(

                "results",

                []

            )

            if len(results) == 0:

                break

            for item in results:

                record = self.parse_item(item)

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

            page += 1

            time.sleep(

                self.delay

            )


        return collected


        ###########################################################
    # Helper Functions
    ###########################################################

    def clean_text(self, text):
        """
        Clean text returned by the LOC API.
        """

        if text is None:
            return ""

        return str(text).strip()

    ###########################################################

    def get_year(self, item):
        """
        Extract year using the shared YearParser.
        """

        fields = [

            item.get("date"),

            item.get("created_published"),

            item.get("item", ""),

            item.get("title"),

            item.get("description")

        ]

        for value in fields:

            if value is None:
                continue

            if isinstance(value, list):
                value = " ".join(str(v) for v in value)

            year = self.year_parser.parse(str(value))

            if year is not None:
                return year

        return None

    ###########################################################

    def extract_image(self, item):
        """
        Get the highest-quality image URL.
        """

        urls = item.get("image_url", [])

        if isinstance(urls, list) and len(urls) > 0:

            return urls[-1]

        return None

    ###########################################################

    def extract_license(self, item):

        fields = [

            "rights",

            "access_restricted",

            "copyright"

        ]

        for field in fields:

            value = item.get(field)

            if value:

                return self.clean_text(value)

        return "Unknown"

    ###########################################################

    def extract_country(self, item):

        country = item.get("location")

        if isinstance(country, list):

            if len(country):

                return self.clean_text(country[0])

        if isinstance(country, str):

            return self.clean_text(country)

        return "United States"

    ###########################################################

    def is_photograph(self, item):

        text = ""

        for key in [

            "title",

            "description",

            "subject"

        ]:

            value = item.get(key)

            if value is None:
                continue

            if isinstance(value, list):

                value = " ".join(value)

            text += " " + str(value)

        text = text.lower()

        blocked = [

            "painting",

            "drawing",

            "map",

            "poster",

            "advertisement",

            "book",

            "newspaper",

            "manuscript",

            "coin",

            "stamp",

            "sculpture",

            "lithograph",

            "engraving"

        ]

        for word in blocked:

            if word in text:

                return False

        return True

    ###########################################################

    def normalize_record(self, record):

        return {

            "id": str(record["id"]),

            "title": record["title"],

            "year": record["year"],

            "image_url": record["image_url"],

            "source": "Library of Congress",

            "license": record["license"],

            "country": record["country"],

            "type": "IMAGE"

        }

    ###########################################################

    def parse_item(self, item):

        image_url = self.extract_image(item)

        if image_url is None:

            return None

        if not self.is_photograph(item):

            return None

        record = {

            "id": item.get("id", ""),

            "title": self.clean_text(item.get("title", "")),

            "year": self.get_year(item),

            "image_url": image_url,

            "license": self.extract_license(item),

            "country": self.extract_country(item)

        }

        return self.normalize_record(record)