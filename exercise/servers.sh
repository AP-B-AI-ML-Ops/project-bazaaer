mlflow server --backend-store-uri sqlite:///mlflow.db &
prefect server start &
wait
echo "All services started"