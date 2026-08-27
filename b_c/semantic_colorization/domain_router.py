"""
Domain Router for Cross-Domain Image Colorization

Routes an input image to the appropriate colorization pipeline
based on the selected domain.
"""

class DomainRouter:

    SUPPORTED_DOMAINS = {
        "historical",
        "bw_photo",
        "sketch",
        "infrared",
    }

    def __init__(self, historical_colorizer=None):
        """
        historical_colorizer:
            Your existing RealtimeColorizer instance.
        """

        self.historical_colorizer = historical_colorizer

        # These will be implemented later
        self.bw_colorizer = None
        self.sketch_colorizer = None
        self.infrared_colorizer = None

    def colorize(self, image, domain):

        domain = domain.lower().strip()

        if domain not in self.SUPPORTED_DOMAINS:
            raise ValueError(
                f"Unsupported domain: {domain}. "
                f"Choose from: {sorted(self.SUPPORTED_DOMAINS)}"
            )

        # -------------------------------------------------
        # HISTORICAL
        # -------------------------------------------------

        if domain == "historical":

            if self.historical_colorizer is None:
                raise RuntimeError(
                    "Historical colorizer has not been initialized."
                )

            # Your existing pipeline
            return self.historical_colorizer.process_frame(image)

        # -------------------------------------------------
        # B&W PHOTO
        # -------------------------------------------------

        if domain == "bw_photo":

            if self.bw_colorizer is None:
                raise NotImplementedError(
                    "B&W Photo colorizer is not implemented yet."
                )

            return self.bw_colorizer.colorize(image)

        # -------------------------------------------------
        # SKETCH
        # -------------------------------------------------

        if domain == "sketch":

            if self.sketch_colorizer is None:
                raise NotImplementedError(
                    "Sketch colorizer is not implemented yet."
                )

            return self.sketch_colorizer.colorize(image)

        # -------------------------------------------------
        # INFRARED
        # -------------------------------------------------

        if domain == "infrared":

            if self.infrared_colorizer is None:
                raise NotImplementedError(
                    "Infrared colorizer is not implemented yet."
                )

            return self.infrared_colorizer.colorize(image)

    # -----------------------------------------------------
    # Register additional domain colorizers
    # -----------------------------------------------------

    def set_bw_colorizer(self, colorizer):
        self.bw_colorizer = colorizer

    def set_sketch_colorizer(self, colorizer):
        self.sketch_colorizer = colorizer

    def set_infrared_colorizer(self, colorizer):
        self.infrared_colorizer = colorizer