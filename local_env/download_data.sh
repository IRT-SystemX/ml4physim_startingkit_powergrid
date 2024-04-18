docker kill download_powergrid_dataset
docker rm download_powergrid_dataset
docker run -it --rm --name=download_powergrid_dataset \
  -v ./program:/app/program \
  -v ./data:/app/data \
  -w /app/program \
  -e PYTHONUNBUFFERED=1 \
  lipsbenchmark/ml4physimpowergrid:latest python /app/program/download_dataset.py
