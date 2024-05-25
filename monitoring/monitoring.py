import pandas as pd
import mlflow
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import os
from dotenv import load_dotenv
import os
import joblib
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently import ColumnMapping

import psycopg

NUMERICAL = [
    "mean_r",
    "mean_g",
    "mean_b",
    "height",
    "width",
    "label"
]


COL_MAPPING = ColumnMapping(
    prediction='label',
    numerical_features=NUMERICAL,
    target=None
)

load_dotenv()

# host, port, user, password
CONNECT_STRING = f'host={os.getenv("POSTGRES_HOST")} port={os.getenv("POSTGRES_PORT")} user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASSWORD")}'


mlflow.set_tracking_uri("http://127.0.0.1:5000")


def prep_db():
    create_table_query = """
    DROP TABLE IF EXISTS metrics;
    CREATE TABLE metrics(
        timestamp timestamp,
        prediction_drift float,
        num_drifted_columns integer,
        share_missing_values float
    );
    """

    with psycopg.connect(CONNECT_STRING, autocommit=True) as conn:
        # zoek naar database genaamd 'test' in de metadata van postgres
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test;")
        with psycopg.connect(f'{CONNECT_STRING} dbname=test') as conn:
            conn.execute(create_table_query)


def prep_data(output_dir: str):
    def calculate_dataset_statistics(dataset):
        stats = []
        for batch in tqdm(dataset):
            images, labels = batch
            for img, lbl in zip(images, labels):
                if lbl.numpy() == "":
                    continue
                img_array = img.numpy()
                mean = np.mean(img_array, axis=(0, 1))
                height, width, _ = img_array.shape
                stats.append({
                    'mean_r': mean[0],
                    'mean_g': mean[1],
                    'mean_b': mean[2],
                    'height': height,
                    'width': width,
                    'label': lbl.numpy()
                })
        return pd.DataFrame(stats)

    train_ds = tf.data.Dataset.load(os.path.join(output_dir, 'train'))
    train_stats = calculate_dataset_statistics(train_ds)
    train_stats.to_csv("train_stats.csv", index=False)

    with psycopg.connect(f'{CONNECT_STRING} dbname=production', autocommit=True) as conn:
        current_stats = conn.execute("SELECT * FROM requests")
        current_stats = pd.DataFrame(current_stats.fetchall())
        current_stats.columns = ['timestamp', 'mean_r', 'mean_g', 'mean_b', 'height', 'width', 'label']
        current_stats.to_csv("current_stats.csv", index=False)
    
    return train_stats, current_stats


def calculate_metrics(current_data, ref_data):

    report = Report(metrics = [
        ColumnDriftMetric(column_name='label'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    report.run(
        reference_data=ref_data,
        current_data=current_data,
        column_mapping=COL_MAPPING
    )

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_cols = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_vals = result['metrics'][2]['result']['current']['share_of_missing_values']

    return prediction_drift, num_drifted_cols, share_missing_vals

def save_metrics_to_db(cursor, date, prediction_drift, num_drifted_cols, share_missing_vals):
    cursor.execute("""
    INSERT INTO metrics(
        timestamp,
        prediction_drift,
        num_drifted_columns,
        share_missing_values
    )
    VALUES (%s, %s, %s, %s);
    """, (date, prediction_drift, num_drifted_cols, share_missing_vals))

def monitor():

    prep_db()
    ref_data, raw_data = prep_data("../data/animal_data_preprocessed")

    start_date = raw_data['timestamp'].min().date()
    
    end_date = raw_data['timestamp'].max().date()

    print(f"Monitoring from {start_date} to {end_date}")


    with psycopg.connect(f'{CONNECT_STRING} dbname=test') as conn:
        with conn.cursor() as cursor:
            for date in pd.date_range(start_date, end_date):
                try:
                    current_data = raw_data[raw_data['timestamp'].dt.date == date.date()]
                    current_data = current_data.drop(columns=['timestamp'])
                    print(ref_data['label'])
                    prediction_drift, num_drifted_cols, share_missing_vals = calculate_metrics(current_data, ref_data)
                    save_metrics_to_db(cursor, date, prediction_drift, num_drifted_cols, share_missing_vals)
                except Exception as e:
                    print(f"Error for date {date}: {e}")
                    continue


if __name__ == "__main__":
    monitor()
