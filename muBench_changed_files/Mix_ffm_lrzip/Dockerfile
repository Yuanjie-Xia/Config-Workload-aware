# Use an official Ubuntu as a base image
# FROM ubuntu:20.04
FROM loongspeed/microservice_v4-screen:mp-v10

# Set environment variables to non-interactive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# --- only change: use archive.debian.org for EOL buster + disable Valid-Until ---
RUN printf '%s\n' \
    'deb [check-valid-until=no] http://archive.debian.org/debian buster main contrib non-free' \
    'deb [check-valid-until=no] http://archive.debian.org/debian buster-updates main contrib non-free' \
    'deb [check-valid-until=no] http://archive.debian.org/debian-security buster/updates main contrib non-free' \
    > /etc/apt/sources.list \
 && apt-get -o Acquire::Check-Valid-Until=false update -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    yasm \
    nasm \
    git \
    pkg-config \
    libtool \
    autoconf \
    automake \
    g++ \
    wget \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libopus-dev \
    libfdk-aac-dev \
    ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir psutil

# Clone FFmpeg repository
RUN git clone https://git.ffmpeg.org/ffmpeg.git /ffmpeg

# Set the working directory
WORKDIR /ffmpeg

# Configure, build, and install
RUN ./configure --enable-gpl --enable-nonfree --enable-libx264 --enable-libx265 --enable-libfdk-aac --enable-libopus --enable-shared && \
    make -j$(nproc) && \
    make install

# Clean up unnecessary files to reduce the image size
RUN rm -rf /ffmpeg

ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
RUN ldconfig

# Set the working directory
WORKDIR /app

# Download test cases
RUN wget https://media.xiph.org/video/derf/y4m/blue_sky_1080p25.y4m

RUN mkdir -p chunks && \
    ffmpeg -i blue_sky_1080p25.y4m -c copy -map 0 -segment_time 0.1 -f segment chunks/blue_chunk_%02d.y4m


# Install dependencies and lrzip
RUN apt-get update && apt-get install -y wget lrzip tar && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir psutil

# Create a test file
RUN mkdir -p /app/lrzip_chunks
COPY chunk_* /app/lrzip_chunks/

# Set the default command to run when the container starts
CMD ["/bin/bash", "/app/start-mp.sh"]
