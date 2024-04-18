docker kill reproduce_baseline
docker rm reproduce_baseline
docker run -it --rm --name=reproduce_baseline \
  -v ./program:/app/program \
  -v ./ingested_program:/app/ingested_program \
  -v ./output:/app/output \
  -v ./data/datasets:/app/data \
  -v ./data/data_grid2op:/root/data_grid2op \
  -w /app/program \
  -e PYTHONUNBUFFERED=1 \
  lipsbenchmark/ml4physimpowergrid:latest python /app/program/reproduce_baseline.py
