docker kill submission_test
docker rm submission_test
docker run --name=submission_test -t -d \
  -v ./program:/app/program \
  -v ./ingested_program:/app/ingested_program \
  -v ./output:/app/output \
  -v ./data/datasets:/app/data \
  -v ./data/data_grid2op:/root/data_grid2op \
  -w /app/program \
  -e PYTHONUNBUFFERED=1 \
  lipsbenchmark/ml4physimpowergrid:latest 
 docker exec -it submission_test bash


