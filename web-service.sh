docker build -t web-service -f web-service/Dockerfile .
docker run --rm -it --network host -e POSTGRES_HOST=localhost -e POSTGRES_PORT=5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password web-service