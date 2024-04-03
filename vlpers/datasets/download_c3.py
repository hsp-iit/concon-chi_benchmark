from pyDataverse.api import NativeApi, DataAccessApi
from pathlib import Path
from git import Repo
from tqdm import tqdm
import requests
from zipfile import ZipFile
import os
from vlpers.utils.logging import logger

base_url = 'https://dataverse.iit.it'
DOI = "doi:10.48557/QJ1166"


def download_file(filename, file_id):
    filename.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://dataverse.iit.it/api/access/datafile/{file_id}'
    response = requests.get(url, params={'User-Agent': 'pydataverse'}, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    id = logger.progress(description=f'[red]Downnloading {filename.name}...', total=total_size // 1000)
    with filename.open("wb") as file:
        for data in response.iter_content(block_size):
            logger.progress(id, advance=len(data) / 1000)
            file.write(data)

def unzip_file(filename):
     with ZipFile(file=filename) as zip_file:
        id = logger.progress(description=f'[red]Extracting {filename.name}...',
                                total=len(zip_file.namelist()))
        for file in zip_file.namelist():
            zip_file.extract(member=file, path=filename.parent)
            logger.progress(id)
        os.remove(filename)

def download_dataset():
    
    # Find root and create dataset dir
    project_root = Repo(search_parent_directories=True).working_dir
    dataset_root = Path(project_root) / "datasets" / "ConCon-Chi"

    dataset_root.mkdir(parents=True, exist_ok=True)

    # Connect to dataverse and get dataset info
    api = NativeApi(base_url)
    dataset = api.get_dataset(DOI)
    files_list = dataset.json()['data']['latestVersion']['files']

    # Download and extract
    for file in files_list:
        filename = dataset_root / file.get('directoryLabel', '') /  file["label"]
        file_id = file["dataFile"]["id"]

        if 'visualization' in filename.as_posix():
            continue

        download_file(filename, file_id)

        if filename.suffix == '.zip':
            unzip_file(filename)

if __name__ == '__main__':
    download_dataset()  