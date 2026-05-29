from icecube_data_reader.downloader import IceCubeData, I3_10

import pytest
data = IceCubeData()

@pytest.mark.skip("convenience")
def test_file_download(output_directory):
    data.fetch(I3_10, write_to=output_directory)
