from prefect import flow
from load.fetch import dataset_download_flow
from load.prep import preprocess_data_flow
from train.hpo import hyperparameter_optimization_flow
from train.train import model_training_flow
import mlflow

DATASET_NAME = "likhon148/animal-data"
DATASET_PATH = "../data"


@flow
def main_flow():
    print("start main flow")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    needs_download = dataset_download_flow(DATASET_NAME, DATASET_PATH)
    train_ds, val_ds = preprocess_data_flow(data_dir=f"{DATASET_PATH}/animal_data", output_dir=f"{DATASET_PATH}/animal_data_preprocessed", img_height=224, img_width=224, batch_size=32, needs_download=needs_download)
    hyperparameter_optimization_flow(train_ds=train_ds, val_ds=val_ds, train_sample_size=500, val_sample_size=100, output_dir="../data/animal_data_preprocessed")
    model_training_flow(output_dir="../data/animal_data_preprocessed", train_ds=train_ds, val_ds=val_ds)


if __name__ == "__main__":
    main_flow()