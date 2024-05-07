import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras_tuner.tuners import RandomSearch
from prefect import task, flow
import os
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.data.experimental import load

hyperparameters = {
    'l2_reg': [0.01, 0.001, 0.0001],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [1e-3, 1e-4, 1e-5]
}

def build_model(hp, input_shape):
    """Builds and compiles a CNN model based on hyperparameter choices."""
    model = Sequential([
        layers.InputLayer(input_shape=input_shape),
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
        layers.Dense(15, activation='softmax')  # Assuming 15 classes
    ])
    
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=hyperparameters['learning_rate'])),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    return model

tuner = RandomSearch(
    lambda hp: build_model(hp, input_shape=(224, 224, 3)),  
    objective='val_accuracy',
    max_trials=1,  
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='animal_classification_tuning',
    overwrite=True
)

@task
def load_datasets(output_dir: str):
    train_ds = load(os.path.join(output_dir, 'train'))
    val_ds = load(os.path.join(output_dir, 'val'))
    return train_ds, val_ds

@task
def random_sample(dataset, sample_size):
    return dataset.shuffle(buffer_size=1024).take(sample_size)

@task
def perform_hyperparameter_optimization(train_ds, val_ds):
    with mlflow.start_run():
        tuner.search(train_ds, validation_data=val_ds, epochs=10, 
                     callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
        # Log top trials
        best_trials = tuner.oracle.get_best_trials(num_trials=5)  
        for trial in best_trials:
            # Log hyperparameters
            mlflow.log_params(trial.hyperparameters.values)

            # Log metrics
            val_accuracy = trial.metrics.get_last_value("val_accuracy")
            print(val_accuracy)
            val_loss = trial.metrics.get_last_value("val_loss")
            print(val_loss)

            if val_accuracy is not None:
                mlflow.log_metric('val_accuracy', val_accuracy)
            if val_loss is not None:
                mlflow.log_metric('val_loss', val_loss)


@flow
def hyperparameter_optimization_flow(train_ds=None, val_ds=None, train_sample_size=500, val_sample_size=100, output_dir="../data/animal_data_preprocessed"):
    if train_ds is None or val_ds is None:
        train_ds, val_ds = load_datasets(output_dir)

    mlflow.set_experiment("Hyperparameter_Optimization")
    
    # Take random samples of the datasets
    sampled_train_ds = random_sample(train_ds, train_sample_size)
    sampled_val_ds = random_sample(val_ds, val_sample_size)
    
    # Perform hyperparameter optimization
    perform_hyperparameter_optimization(sampled_train_ds, sampled_val_ds)
