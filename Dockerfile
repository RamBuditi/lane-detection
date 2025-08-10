# 1. Start from an NVIDIA base image compatible with CUDA 12.1
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# 3. Install system dependencies and Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda to /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# 4. Set the PATH to include Conda
ENV PATH /opt/conda/bin:$PATH

# 5. Forcefully re-configure Conda's base channels
RUN conda config --system --set channel_priority strict && \
    conda config --system --remove channels defaults && \
    conda config --system --add channels conda-forge

# 6. Set the working directory
WORKDIR /app

# 7. HYBRID INSTALL PART 1: Create the core Conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# 8. Set the default shell for subsequent build commands
SHELL ["conda", "run", "-n", "lane-detection", "/bin/bash", "-c"]

# 9. HYBRID INSTALL PART 2: Install MLOps packages with pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# 10. Create a non-root user that matches the host's UID/GID
ARG UID
ARG GID
RUN groupadd -g $GID -o shadowz && \
    useradd -m -u $UID -g $GID -s /bin/bash shadowz

# 11. Switch to the new user and initialize conda FOR THEM
USER shadowz
RUN conda init bash && \
    echo "conda activate lane-detection" >> /home/shadowz/.bashrc

# 12. Switch back to root user for final operations
USER root

# 13. Copy all your project code into the container
# Copying as root ensures files are owned by root, then container starts as non-root
COPY . .

# 14. Set ownership of the app directory to the non-root user
RUN chown -R shadowz:shadowz /app

# 15. Expose a port and set a default command
EXPOSE 8000
CMD ["tail", "-f", "/dev/null"]