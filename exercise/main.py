from prefect import flow
from fetch.fetch import dataset_download_flow

DATASET_NAME = "likhon148/animal-data"
DATASET_PATH = "../data"

@flow
def main_flow():
    dataset_download_flow(DATASET_NAME, DATASET_PATH)


if __name__ == "__main__":
    main_flow()