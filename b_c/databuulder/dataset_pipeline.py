from apis.europeana import EuropeanaAPI
from apis.loc import LOCAPI
from apis.wikimedia import WikimediaAPI

from downloader.metadata import MetadataManager
from downloader.downloader import ImageDownloader

from filters.photograph import PhotographFilter
from filters.duplicate import DuplicateFilter
from filters.quality import QualityFilter

from utils.logger import DatasetLogger

from config import *

logger = DatasetLogger()


def main():

    logger.info("Starting Dataset Collection")

    #######################################################
    # Collect Metadata
    #######################################################

    europeana = EuropeanaAPI(
        api_key=EUROPEANA_API_KEY,
        max_records=MAX_IMAGES
    )

    loc = LOCAPI(
        max_records=MAX_IMAGES
    )

    wiki = WikimediaAPI(
        max_records=MAX_IMAGES
    )

    metadata = []

    metadata.extend(europeana.collect())

    metadata.extend(loc.collect())

    metadata.extend(wiki.collect())

    #######################################################
    # Remove duplicates
    #######################################################

    manager = MetadataManager()

    metadata = manager.remove_duplicate_metadata(metadata)

    manager.save_csv(metadata)

    manager.statistics(metadata)

    #######################################################
    # Photograph filtering
    #######################################################

    photo_filter = PhotographFilter()

    metadata = photo_filter.filter(metadata)

    #######################################################
    # Download
    #######################################################

    downloader = ImageDownloader()

    downloader.download(metadata)

    #######################################################
    # Quality filtering
    #######################################################

    quality = QualityFilter()

    quality.run(DATASET_DIR)

    #######################################################
    # Duplicate filtering
    #######################################################

    duplicate = DuplicateFilter()

    duplicate.run(DATASET_DIR)

    logger.info("Finished Successfully")


if __name__ == "__main__":

    main()