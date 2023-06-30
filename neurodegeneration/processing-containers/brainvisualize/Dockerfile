FROM --platform=linux/amd64 kitware/vtk:ci-el7-mindeps-20230216
FROM --platform=linux/amd64 python:3.9-slim-buster

LABEL IMAGE="brainvis"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="True"

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libgl1 libglib2.0-0 xvfb

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN mkdir /src 

COPY files/reportdriver_generateBrainVisual.py /src
COPY files/generateBrainVisual.py /src
COPY files/createcmap.py /src

RUN mkdir /refs

COPY refs/ /refs

CMD ["python3","-u","src/reportdriver_generateBrainVisual.py"]
