FROM ubuntu:18.04
LABEL maintainer="acw6ze@virginia.edu"

ARG AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ARG NN_VERSION_NUMBER=$NN_VERSION_NUMBER

ENV NN_VERSION_NUMBER=$NN_VERSION_NUMBER

RUN apt-get update
RUN apt-get install -y sudo
RUN apt-get install -y git

RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER root

SHELL ["/bin/bash", "-c"]

RUN sudo apt-get -qq install curl vim git zip
RUN sudo curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN sudo unzip awscliv2.zip
RUN sudo ./aws/install
RUN echo ${AWS_ACCESS_KEY_ID}
RUN aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
RUN aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
RUN aws configure set region us-east-1



ENV APP_LOC /home/ubuntu/Neural_Network

RUN mkdir -p $APP_LOC

WORKDIR /home/ubuntu/Neural_Network

RUN sudo mkdir /home/project
COPY Neural_Network /home/ubuntu/Neural_Network
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash ./Miniconda3-latest-Linux-x86_64.sh -b
ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda --version
RUN conda update -n base conda
RUN conda --version
RUN conda install -n base conda-libmamba-solver
RUN conda config --set ssl_verify no
RUN conda init bash
RUN conda config --set auto_activate_base false
RUN conda create -n BASNET python=3.8 --yes
RUN echo ". /root/miniconda3/etc/profile.d/conda.sh" >> /root/.profile
RUN echo "conda activate BASNET" >> /root/.bashrc
RUN conda run -n BASNET conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda run -n BASNET pip install panda
RUN conda run -n BASNET pip install pandas
RUN conda run -n BASNET pip install openpyxl
RUN conda run -n BASNET pip install numpy
RUN conda run -n BASNET pip install scikit-learn
RUN conda run -n BASNET pip install scipy
RUN conda run -n BASNET pip install tensorflow
RUN conda run -n BASNET pip install tensorboard
RUN conda run -n BASNET pip install tensorboard-logger
RUN conda run -n BASNET pip install tensorflow-estimator
RUN conda run -n BASNET pip install tqdm
RUN conda run -n BASNET pip install opencv-python-headless
RUN conda run -n BASNET pip install requests
RUN sudo apt-get install -y libgl1-mesa-dev
RUN sudo aws s3 cp s3://aiai-pipeline-files/BaSNet_model_best.pkl /home/ubuntu/Neural_Network/models/BaSNet_model_best.pkl
RUN chmod -R a+rwx /home/project
RUN chmod +x /home/ubuntu/Neural_Network/run.sh
SHELL ["conda", "run", "-n", "BASNET", "/bin/bash"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "BASNET", "/bin/bash", "-c", "tail -f /dev/null"]
