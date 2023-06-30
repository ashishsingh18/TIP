FROM --platform=linux/amd64 python:3.9-slim-buster

LABEL IMAGE="BIONORM"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="True"

RUN apt-get update
RUN apt-get -y install libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info libgl1

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN mkdir /src 

COPY files/normative_biomarker_visualizer.py /src
COPY files/reportdriver_normative_biomarker_visualizer.py /src
COPY files/spareAD.py /src
COPY files/spareBA.py /src
COPY files/wmls.py /src
COPY files/createcmap.py /src
COPY files/boxoffplot.py /src

RUN mkdir /refs
COPY refs/ /refs

CMD ["python3","-u","src/reportdriver_normative_biomarker_visualizer.py"]
