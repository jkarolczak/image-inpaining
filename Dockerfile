FROM anibali/pytorch:1.10.0-nocuda-ubuntu20.04

#RUN conda install -c conda-forge opencv pandas matplotlib scikit-learn yaml
#RUN conda update -n base -c conda-forge opencv pandas matplotlib scikit-learn yaml

USER root
RUN apt-get update
RUN apt-get install -y python3-opencv
USER user

RUN pip install opencv-python pandas matplotlib scikit-learn
RUN pip install pyyaml

COPY ./docker/docker_infer.py /app/infer.py
COPY ./docker/generator_model.pt /app/generator_model.pt

RUN mkdir /app/src
COPY ./src /app/src

# Set the default command to python3
CMD ["python3"]