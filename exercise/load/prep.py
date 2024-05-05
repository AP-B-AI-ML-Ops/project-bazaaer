from prefect import task, flow
from tensorflow.keras import layers

import tensorflow as tf
import os


@task
def load_datasets(data_dir: str, img_height: int, img_width: int, batch_size: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds


@task
def prepare_datasets(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


@task
def normalize_datasets(train_ds, val_ds):
    normalization_layer = layers.Rescaling(1./255)
    train_ds_norm = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds_norm = val_ds.map(lambda x, y: (normalization_layer(x), y))
    return train_ds_norm, val_ds_norm


@task
def save_datasets(train_ds_norm, val_ds_norm, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    tf.data.experimental.save(train_ds_norm, os.path.join(output_dir, 'train'))
    tf.data.experimental.save(val_ds_norm, os.path.join(output_dir, 'val'))
    return train_ds_norm, val_ds_norm


@flow
def preprocess_data_flow(data_dir: str = "../data/animal_data", 
                         img_height: int = 224, 
                         img_width: int = 224, 
                         batch_size: int = 32,
                         output_dir: str = "../data/animal_data_preprocessed"):
    train_ds, val_ds = load_datasets(data_dir, img_height, img_width, batch_size)
    train_ds, val_ds = prepare_datasets(train_ds, val_ds)
    train_ds_norm, val_ds_norm = normalize_datasets(train_ds, val_ds)
    train_ds_norm, val_ds_norm = save_datasets(train_ds_norm, val_ds_norm, output_dir)
    return train_ds_norm, val_ds_norm
