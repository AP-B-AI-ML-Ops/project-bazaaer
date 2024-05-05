from prefect import flow
from fetch import kaggle_dataset_downloader

DATASET_NAME = "likhon148/animal-data"
DATASET_PATH = "../data"

@flow
def main_flow():
    kaggle_dataset_downloader(DATASET_NAME, DATASET_PATH)


if __name__ == "__main__":
    main_flow()