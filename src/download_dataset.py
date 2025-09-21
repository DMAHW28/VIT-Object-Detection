import os
import keras
import shutil

path_to_downloaded_file = keras.utils.get_file(
    fname="caltech_101_zipped",
    origin="https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip",
    extract=True,
    archive_format="zip",  # downloaded file format
    cache_dir="/",  # cache and extract in current directory
)
download_base_dir = os.path.dirname(path_to_downloaded_file)
# Extracting tar files found inside main zip file
shutil.unpack_archive(os.path.join(download_base_dir, "caltech_101_zipped", "caltech-101", "101_ObjectCategories.tar.gz"), ".")
shutil.unpack_archive(os.path.join(download_base_dir, "caltech_101_zipped", "caltech-101", "Annotations.tar"), ".")