sudo docker compose down ml-service ml-qdrant minio articles-service postgres-articles
sudo docker volume rm atllm_project_qdrant-data atllm_project_minio-data atllm_project_postgres-articles-data
sudo docker compose up -d ml-service ml-qdrant minio articles-service postgres-articles