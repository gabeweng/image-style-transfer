FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    tmux \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone diffusers examples so training scripts are available
RUN git clone --depth 1 https://github.com/huggingface/diffusers /tmp/diffusers && \
    cp -r /tmp/diffusers/examples /workspace/diffusers_examples && \
    pip install --no-cache-dir /tmp/diffusers && \
    rm -rf /tmp/diffusers

COPY . /workspace/

# Default: open a shell; override with the training command in tmux
CMD ["/bin/bash"]
