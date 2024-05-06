import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Initialize MLflow
mlflow.set_experiment("animal_classification")

# Start a new MLflow run
mlflow.tensorflow.autolog()

hyperparameters = {
    'l2_reg': [.1,.01,.001],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [1e-3, 5e-4, 1e-4]
}

def build_model(hp):
    model = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu',
                      kernel_regularizer=l2(hp.Choice('l2_reg', values=hyperparameters['l2_reg']))),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu',
                      kernel_regularizer=l2(hp.Choice('l2_reg', values=hyperparameters['l2_reg']))),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu',
                      kernel_regularizer=l2(hp.Choice('l2_reg', values=hyperparameters['l2_reg']))),
        layers.MaxPooling2D(),
        layers.Dropout(hp.Choice('dropout_rate', values=hyperparameters['dropout_rate'])),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(15)
    ])
    
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=hyperparameters['learning_rate'])),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    project_name='animal_classification'
)

stop_early = EarlyStopping(monitor='val_loss', patience=5)

with mlflow.start_run():
    tuner.search(train_ds, validation_data=val_ds, epochs=10, callbacks=[stop_early])

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build and train the best model
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[stop_early])
    
    # Log the best hyperparameters
    mlflow.log_params(best_hps.values)
    
    # Log the best model
    mlflow.keras.log_model(best_model, "best_model")
