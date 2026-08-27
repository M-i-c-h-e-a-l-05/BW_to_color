"""
filters/photograph.py

Photograph Metadata Filter

Author: Micheal Leveiro
"""


class PhotographFilter:

    def __init__(self):

        self.allowed = {

            "IMAGE",

            "PHOTOGRAPH",

            "PHOTO",

            "NEGATIVE",

            "GLASS NEGATIVE",

            "DIGITAL IMAGE"

        }

        self.blocked = {

            "PAINTING",

            "DRAWING",

            "MAP",

            "BOOK",

            "MANUSCRIPT",

            "NEWSPAPER",

            "POSTER",

            "ADVERTISEMENT",

            "COIN",

            "STAMP",

            "SCULPTURE",

            "PRINT"

        }

    #######################################################

    def filter(self, metadata):

        filtered = []

        for record in metadata:

            record_type = str(

                record.get("type", "")

            ).upper()

            title = str(

                record.get("title", "")

            ).upper()

            # Remove blocked types
            if record_type in self.blocked:
                continue

            # Remove if title contains blocked words
            blocked = False

            for word in self.blocked:

                if word in title:

                    blocked = True
                    break

            if blocked:
                continue

            # Keep known photograph types
            if record_type in self.allowed:

                filtered.append(record)

                continue

            # Unknown type → keep for now
            filtered.append(record)

        print(
            f"Photograph Filter: {len(filtered)} images kept"
        )

        return filtered