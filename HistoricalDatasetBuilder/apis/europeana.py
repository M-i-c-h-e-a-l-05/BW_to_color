"""
apis/europeana.py

Improved Europeana API Client
Research Version

Author: Micheal Leveiro
"""

import time
import requests

from utils.year_parser import YearParser


class EuropeanaAPI:

    BASE_URL = "https://api.europeana.eu/record/v2/search.json"

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
        # "museum",
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
        # "festival",
        # "archive",
        # "black and white photograph",
        # "vintage photograph"
    ]

    ############################################################

    def __init__(

        self,

        api_key,

        rows=100,

        max_records=750,

        delay=0.5,

        retries=5

    ):

        self.api_key = api_key

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

    ############################################################

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

    ############################################################

    def collect(self):

        records = []

        seen = set()

        for term in self.SEARCH_TERMS:

            print(f"\nSearching Europeana: {term}")

            new_records = self.search_term(

                term,

                seen

            )

            records.extend(new_records)

            print(

                f"Total Europeana records: {len(records)}"

            )

            if len(records) >= self.max_records:

                break

        return records[:self.max_records]

    ############################################################

    def search_term(

        self,

        term,

        seen

    ):

        collected = []

        start = 1

        while True:

            params = {

                "wskey": self.api_key,

                "query": term,

                "media": "true",

                "reusability": "open",

                "profile": "rich",

                "rows": self.rows,

                "start": start

            }

            data = self.request_with_retry(params)

            if data is None:

                break

            items = data.get(

                "items",

                []

            )

            if len(items) == 0:

                break

            for item in items:

                record = self.parse_item(item)

                if record is None:

                    continue

                if record["id"] in seen:

                    continue

                seen.add(record["id"])

                collected.append(record)

            print(

                f"{term}: {len(collected)} images"

            )

            start += self.rows

            time.sleep(self.delay)

        return collected


        ############################################################
    # Helper Functions
    ############################################################

    def clean_html(self, text):
        """
        Remove HTML tags and HTML entities.
        """

        if text is None:
            return ""

        import re
        import html

        text = re.sub(r"<[^>]+>", "", str(text))
        text = html.unescape(text)

        return text.strip()

    ############################################################

    def get_first(self, value):

        if isinstance(value, list):

            if len(value):

                return value[0]

            return ""

        return value if value is not None else ""

    ############################################################

    def get_year(self, item):
        """
        Extract year using the shared YearParser.
        """

        fields = [

            item.get("year"),

            item.get("timestamp"),

            item.get("title"),

            item.get("dcDate"),

            item.get("edmTimespanLabel"),

            item.get("date"),

            item.get("description")

        ]

        for value in fields:

            if value is None:
                continue

            value = self.get_first(value)

            value = self.clean_html(value)

            year = self.year_parser.parse(value)

            if year is not None:

                return year

        return None

    ############################################################

    def extract_license(self, item):

        fields = [

            "rights",

            "edmRights",

            "license"

        ]

        for field in fields:

            value = item.get(field)

            if value is None:
                continue

            value = self.get_first(value)

            value = self.clean_html(value)

            if value != "":

                return value

        return "Unknown"

    ############################################################

    def extract_country(self, item):

        fields = [

            "country",

            "dataProviderCountry",

            "provider"

        ]

        for field in fields:

            value = item.get(field)

            if value is None:
                continue

            value = self.get_first(value)

            value = self.clean_html(value)

            if value != "":

                return value

        return ""

    ############################################################

    def is_photograph(self, item):
        """
        Remove obvious non-photographs.
        """

        text = ""

        fields = [

            "title",

            "type",

            "dcType",

            "description"

        ]

        for field in fields:

            value = item.get(field)

            if value is None:
                continue

            value = self.get_first(value)

            text += " " + self.clean_html(value)

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

            "engraving",

            "lithograph",

            "woodcut",

            "illustration",

            "sculpture"

        ]

        for word in blocked:

            if word in text:

                return False

        return True

    ############################################################

    def normalize_record(self, record):

        return {

            "id": str(record["id"]),

            "title": record["title"],

            "year": record["year"],

            "image_url": record["image_url"],

            "source": "Europeana",

            "license": record["license"],

            "country": record["country"],

            "type": "IMAGE"

        }

    ############################################################

    def parse_item(self, item):

        image_url = None

        if "edmPreview" in item:

            preview = item["edmPreview"]

            if isinstance(preview, list) and len(preview):

                image_url = preview[0]

            elif isinstance(preview, str):

                image_url = preview

        if image_url is None:

            return None

        if not self.is_photograph(item):

            return None

        title = self.clean_html(

            self.get_first(

                item.get("title", "")

            )

        )

        record = {

            "id": item.get("id", ""),

            "title": title,

            "year": self.get_year(item),

            "image_url": image_url,

            "license": self.extract_license(item),

            "country": self.extract_country(item)

        }

        return self.normalize_record(record)