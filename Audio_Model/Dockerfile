# Dockerfile for Audio Labeling Script with GitHub Source and CUDA Support

FROM ubuntu:22.04
LABEL maintainer="ekn8kz@virginia.edu"

# --- System Setup ---
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    curl \
    vim \
    zip \
    wget \
    libgl1-mesa-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Create user ---
RUN adduser --disabled-password --gecos '' ubuntu && \
    adduser ubuntu sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu
SHELL ["/bin/bash", "-c"]

# --- Miniconda Installation ---
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/home/ubuntu/miniconda3/bin:$PATH"
RUN echo ". \$HOME/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
RUN conda init bash && conda config --set auto_activate_base false

# --- Conda Environment Setup ---
RUN conda create -y -n audio_env python=3.10.17
SHELL ["conda", "run", "-n", "audio_env", "/bin/bash", "-c"]
RUN pip install --no-cache-dir torch==2.2.2+cu118 \
    torchvision==0.17.2+cu118 \
    torchaudio==2.2.2+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
    transformers==4.51.3 \
    scikit-learn==1.6.1 \
    pandas==2.2.3 \
    numpy==1.24.4 \
    matplotlib==3.10.3

# --- App Setup ---
ENV APP_LOC /home/ubuntu/audio_labeling
RUN mkdir -p $APP_LOC
WORKDIR $APP_LOC

# Clone the script from GitHub and place it in the working directory
RUN git clone https://github.com/Powercoder64/MM_AIAI.git && \
    cp MM_AIAI/Audio_Model/audio_label.py $APP_LOC

# Placeholder: User should mount the model directory during docker run
# e.g., docker run -v $(pwd)/model:/home/ubuntu/audio_labeling/model

# --- Entry Point ---
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "audio_env", "python", "audio_label.py"]
