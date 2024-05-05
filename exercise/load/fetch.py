from prefect import task, flow
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import json


@task
def check_dataset_exists(dataset_path:str):
    exists = os.path.exists(dataset_path + "/animal_data")
    print(f"Dataset exists: {exists}")
    return exists

@task
def authenticate_kaggle():
    api = KaggleApi()
    api.authenticate()
    return api

@task(retries=4, retry_delay_seconds=2)
def fetch_remote_metadata(api, dataset_name:str):
    files = api.dataset_list_files(dataset_name).files
    if files:
        last_updated = max([file.refreshed for file in files])
    else:
        last_updated = 0  # or some other default value or raise an error
    return {"lastUpdated": last_updated}

@task
def fetch_local_metadata(metadata_file:str):
    if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
        with open(metadata_file, "r") as file:
            return json.load(file)
    return None

@task
def compare_metadata(local_metadata, remote_metadata):
    if not local_metadata:
        return True  # No local metadata, needs download
    return local_metadata['lastUpdated'] < remote_metadata['lastUpdated']

@task(retries=4, retry_delay_seconds=2)
def download_dataset(api, dataset_name:str, dataset_path:str, metadata_file:str):
    api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)
    remote_metadata = fetch_remote_metadata(api, dataset_name)
    with open(metadata_file, "w") as file:
        json.dump(remote_metadata, file)

@flow
def dataset_download_flow(dataset_name:str, dataset_path:str):
    metadata_file = os.path.join(dataset_path, "animal_data_metadata.json")

    exists = check_dataset_exists(dataset_path)

    if not exists:
        api = authenticate_kaggle()
        download_dataset(api, dataset_name, dataset_path, metadata_file)
    else:
        api = authenticate_kaggle()
        remote_metadata = fetch_remote_metadata(api, dataset_name)
        local_metadata = fetch_local_metadata(metadata_file)
        needs_download = compare_metadata(local_metadata, remote_metadata)
        print(f"Needs download: {needs_download}")
        
        if needs_download:
            download_dataset(api, dataset_name, dataset_path, metadata_file)