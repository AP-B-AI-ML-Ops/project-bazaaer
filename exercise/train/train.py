import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Retrieve best hyperparameters from MLflow
client = mlflow.tracking.MlflowClient()
experiment_id = client.get_experiment_by_name("Hyperparameter_Optimization").experiment_id
best_run = client.search_runs(
    experiment_ids=[experiment_id], 
    order_by=["metrics.val_accuracy DESC"], 
    max_results=1
)[0]

best_hps = {
    'l2_reg': float(best_run.data.params['l2_reg']),
    'dropout_rate': float(best_run.data.params['dropout_rate']),
    'learning_rate': float(best_run.data.params['learning_rate'])
}

final_model = Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(16, 3, padding='same', activation='relu',
                  kernel_regularizer=l2(best_hps['l2_reg'])),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_regularizer=l2(best_hps['l2_reg'])),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu',
                  kernel_regularizer=l2(best_hps['l2_reg'])),
    layers.MaxPooling2D(),
    layers.Dropout(best_hps['dropout_rate']),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(15)
])

final_model.compile(optimizer=Adam(best_hps['learning_rate']),
                    loss=SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

mlflow.set_experiment("Final_Model_Training")

with mlflow.start_run():
    mlflow.tensorflow.autolog()  # Enable automatic logging

    # Train the model
    final_model.fit(full_train_ds, validation_data=full_val_ds, epochs=10)

    # Log and register the model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, "FinalModel")