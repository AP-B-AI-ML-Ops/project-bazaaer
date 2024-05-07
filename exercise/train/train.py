import os
import mlflow
from prefect import task, flow
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.data.experimental import load as tf_load  # Explicitly alias

# Function to build the final model
def build_final_model(best_hps, input_shape):
    final_model = Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=l2(best_hps['l2_reg'])),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=l2(best_hps['l2_reg'])),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(best_hps['l2_reg'])),
        layers.MaxPooling2D(),
        layers.Dropout(best_hps['dropout_rate']),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(15, activation='softmax')
    ])
    final_model.compile(optimizer=Adam(best_hps['learning_rate']),
                        loss=SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])
    return final_model

# Task to retrieve the best hyperparameters
@task
def retrieve_best_hyperparameters():

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Hyperparameter_Optimization")
    if experiment is None:
        raise ValueError("Experiment 'Hyperparameter_Optimization' does not exist.")
    experiment_id = experiment.experiment_id

    best_run = client.search_runs(
        experiment_ids=[experiment_id], 
        order_by=["metrics.val_accuracy DESC"], 
        max_results=1
    )[0]

    return {
        'l2_reg': float(best_run.data.params['l2_reg']),
        'dropout_rate': float(best_run.data.params['dropout_rate']),
        'learning_rate': float(best_run.data.params['learning_rate'])
    }

# Task to load datasets
@task
def load_datasets(output_dir: str):
    train_ds = tf_load(os.path.join(output_dir, 'train'))
    val_ds = tf_load(os.path.join(output_dir, 'val'))
    return train_ds, val_ds

# Task to execute model training
@task
def execute_model_training(full_train_ds, full_val_ds, hyperparameters):

    final_model = build_final_model(hyperparameters, input_shape=(224, 224, 3))

    mlflow.tensorflow.autolog()
    final_model.fit(full_train_ds, validation_data=full_val_ds, epochs=10)

# Task to register the final model
@task
def register_final_model():
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, "FinalModel")

# Flow for model training
@flow
def model_training_flow(output_dir="../data/animal_data_preprocessed", train_ds=None, val_ds=None):
    mlflow.set_experiment("Final_Model_Training")

    if train_ds is None or val_ds is None:
        train_ds, val_ds = load_datasets(output_dir)
    
    hyperparameters = retrieve_best_hyperparameters()
    with mlflow.start_run():
        execute_model_training(train_ds, val_ds, hyperparameters)
        register_final_model()
