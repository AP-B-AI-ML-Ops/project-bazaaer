from prefect import flow
from load.fetch import dataset_download_flow
from load.prep import preprocess_data_flow
import mlflow

DATASET_NAME = "likhon148/animal-data"
DATASET_PATH = "../data"

@flow
def main_flow():
    print("start main flow")

    dataset_download_flow(DATASET_NAME, DATASET_PATH)
    preprocess_data_flow(data_dir=f"{DATASET_PATH}/animal_data", output_dir=f"{DATASET_PATH}/animal_data_preprocessed", img_height=224, img_width=224, batch_size=32)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

if __name__ == "__main__":
    main_flow()