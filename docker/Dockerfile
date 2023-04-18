FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
#FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
#FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-get update

RUN pip install youtube-dl

RUN apt-get install -y curl git

RUN curl -o /requirements.txt https://raw.githubusercontent.com/jacv050/hyperfuture/main/requirements.txt \
	&& pip install -r /requirements.txt

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install av

RUN pip install neptune-client

RUN curl -o /requirements2.txt https://raw.githubusercontent.com/jacv050/capturing-implicit-hierarchical-structure/master/requirements.txt \
	&& pip install -r /requirements2.txt

RUN pip install opencv-python
