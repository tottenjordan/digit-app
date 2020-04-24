ARG TF_SERVING_VERSION=latest
ARG TF_SERVING_BUILD_IMAGE=tensorflow/serving:${TF_SERVING_VERSION}-devel

FROM ${TF_SERVING_BUILD_IMAGE} as build_image
FROM ubuntu:18.04

# Install git
RUN apt-get update apt-get install -y git
RUN apt-get install -y build-essential python3.8 python3.8-dev python3-pip python3.8-venv

# TO DO: insert github URL
RUN mkdir /digitapp git clone https://github.com/tottenjordan/digit-app/tree/master/app 

# set working directory
WORKDIR /app

# COPY app ./app
# COPY digits .
# COPY static .
# COPY templates .
# COPY appli.py .
# COPY requirements.txt .

#COPY app ./app

RUN pip3 install --no-cache-dir -r app/requirements.txt

# Expose port
EXPOSE 8500
# REST
EXPOSE 8501

# model name should match the docker target path folder
ENV MODEL_NAME=digits

# RUN -p 8501:8501 -e MODEL_NAME=digits tensorflow/serving

CMD ["python3", "./appli.py"]