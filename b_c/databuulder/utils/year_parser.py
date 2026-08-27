"""
utils/year_parser.py

Historical Year Parser

Author: Micheal Leveiro
"""

import re


class YearParser:

    def __init__(self):

        self.min_year = 1800
        self.max_year = 2100

    #########################################################

    def valid(self, year):

        return self.min_year <= year <= self.max_year

    #########################################################

    def parse(self, value):

        if value is None:
            return None

        text = str(value).strip().lower()

        if text == "":
            return None

        # -----------------------------------------
        # YYYY-MM-DD
        # Example: 1944-06-05
        # -----------------------------------------

        m = re.search(r"(18|19|20)\d{2}", text)

        if m:

            year = int(m.group())

            if self.valid(year):
                return year

        # -----------------------------------------
        # 1910-1914
        # Return midpoint
        # -----------------------------------------

        m = re.search(
            r"((18|19|20)\d{2})\D+((18|19|20)\d{2})",
            text
        )

        if m:

            y1 = int(m.group(1))
            y2 = int(m.group(3))

            if self.valid(y1) and self.valid(y2):

                return (y1 + y2) // 2

        # -----------------------------------------
        # between 1901 and 1908
        # -----------------------------------------

        m = re.search(
            r"between\s+((18|19|20)\d{2}).+?((18|19|20)\d{2})",
            text
        )

        if m:

            y1 = int(m.group(1))
            y2 = int(m.group(3))

            return (y1 + y2) // 2

        # -----------------------------------------
        # circa / ca / c.
        # -----------------------------------------

        m = re.search(
            r"(18|19|20)\d{2}",
            text
        )

        if m:

            year = int(m.group())

            if self.valid(year):
                return year

        # -----------------------------------------
        # 1950s
        # -----------------------------------------

        m = re.search(
            r"((18|19|20)\d{2})s",
            text
        )

        if m:

            year = int(m.group(1))

            if self.valid(year):
                return year

        # -----------------------------------------
        # 20th century
        # Approximate to midpoint
        # -----------------------------------------

        m = re.search(
            r"(\d+)(st|nd|rd|th)\s+century",
            text
        )

        if m:

            century = int(m.group(1))

            year = (century - 1) * 100 + 50

            if self.valid(year):
                return year

        # -----------------------------------------
        # Last attempt
        # -----------------------------------------

        years = re.findall(
            r"(18|19|20)\d{2}",
            text
        )

        if years:

            year = int(years[0])

            if self.valid(year):
                return year

        return None

    #########################################################

    def era(self, year):

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