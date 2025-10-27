# Use an official lightweight base image
# FROM ubuntu:latest
# FROM msvcbench/microservice_v4-screen
FROM loongspeed/microservice_v4-screen:latest

# Install dependencies and lrzip
RUN apt-get update && apt-get install -y wget lrzip tar && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir psutil

# Create a test file
RUN mkdir -p /app/chunks
COPY chunk_* /app/chunks/

# Default command (optional)
# CMD ["lrzip", "--help"]
CMD ["/bin/bash", "/app/start-mp.sh"]
