"""
Define all necessary tools to download the two latest IceCube point source data releases.
"""

import numpy as np
from zipfile import ZipFile
import tarfile
from pathlib import Path
from tqdm import tqdm
import requests
import requests_cache
import time
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

data_directory = os.path.abspath(os.path.join(os.path.expanduser("~"), ".icecube_data"))


# I dont like the naming of variables in which only one character is different,
# this is prone to typos in scripts and subsequent bugs
DR1 = "20210126"
DR2 = "IceTracks-DR2"

# Same
I3_14 = DR2
I3_10 = DR1

available_datasets = {
    I3_14: {
        "url": "https://dataverse.harvard.edu/api/access/dataset/:persistentId/versions/2.0?persistentId=doi:10.7910/DVN/MMIIZA",
        "dir": "IceTracksDR2",
        "subdir": "icecube_14year_ps",
    },
    I3_10: {
        "url": "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/VKL316",
        "dir": "20210126_PS-IC40-IC86_VII",
        "subdir": "icecube_10year_ps",
    },
}


class IceCubeData:
    """
    Handle the interface with IceCube's public data
    releases hosted on their website.
    """

    def __init__(
        self,
        data_directory=data_directory,
        cache_name=Path(".cache"),
    ):
        """
        Handle the interface with IceCube's public data
        releases hosted on their website.

        :param base_url: Base url for data releases
        :param data_directory: Where to put the data
        :param cache_name: Name of the requests cache
        """

        self.data_directory = data_directory

        requests_cache.install_cache(
            cache_name=cache_name,
            expire_after=-1,
        )

        # Make data directory if it doesn't exist
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def fetch(self, *datasets, overwrite=False, write_to=None):
        """
        Downloads and unzips the given datasets.

        :param datasets: A list of dataset names
        :param overwrite: Overwrite existing files
        :param write_to: Optional custom location
        """

        if write_to:
            old_dir = self.data_directory

            self.data_directory = write_to

        for dataset in datasets:
            if dataset not in available_datasets:
                raise ValueError("Dataset %s is not in list of known datasets" % dataset)

            ds = available_datasets[dataset]
            url = ds["url"]
            dl_dir = ds["dir"]
            local_path = os.path.join(self.data_directory, dl_dir)
            subdir = ds["subdir"]
            file = os.path.join(local_path, dl_dir + ".zip")
            # Only fetch if not already there!
            if not os.path.exists(local_path) or overwrite:
                os.makedirs(local_path, exist_ok=True)
                # Don't cache this as we want to stream
                with requests_cache.disabled():
                    response = requests.get(url, stream=True)

                    if response.ok:
                        # For progress bar description
                        short_name = dataset
                        if len(dataset) > 40:
                            short_name = dataset[0:40] + "..."

                        # Save locally
                        with (
                            open(file, "wb") as f,
                            tqdm(
                                desc=short_name,
                            ) as bar,
                        ):
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                size = f.write(chunk)
                                bar.update(size)

                        # Unzip
                        if subdir:
                            dataset_dir = os.path.join(local_path, subdir)
                        else:
                            dataset_dir = local_path
                        with ZipFile(file, "r") as zip_ref:
                            zip_ref.extractall(dataset_dir)

                        # Delete zipfile
                        os.remove(file)

                        # Check for further compressed files in the extraction
                        tar_files = find_files(dataset_dir, ".tar")

                        zip_files = find_files(dataset_dir, ".zip")

                        for tf in tar_files:
                            tar = tarfile.open(tf)
                            tar.extractall(os.path.splitext(tf)[0])

                        for zf in zip_files:
                            with ZipFile(zf, "r") as zip_ref:
                                zip_ref.extractall(zf[:-4])

                crawl_delay()

        if write_to:
            self.data_directory = old_dir


def find_files(directory: Path, keyword: str):
    """
    Find files in a directory that contain
    a keyword.

    :param directory: Directory to traverse
    :param keyword: Keyword to search for

    :returns: List of found files
    """

    found_files = []

    for root, dirs, files in os.walk(directory):
        if files:
            for f in files:
                if keyword in f:
                    found_files.append(os.path.join(root, f))

    return found_files


def find_folders(directory: Path, keyword: str):
    """
    Find subfolders in a directory that
    contain a keyword.

    :param directory: Directory to traverse
    :param keyword: Keyword to search for

    :returns: List of directories
    """

    found_folders = []

    for root, dirs, files in os.walk(directory):
        if dirs:
            for d in dirs:
                if keyword in d:
                    found_folders.append(os.path.join(root, d))

    return found_folders


def crawl_delay():
    """
    Delay between sending HTML requests.
    """

    time.sleep(np.random.uniform(5, 10))
