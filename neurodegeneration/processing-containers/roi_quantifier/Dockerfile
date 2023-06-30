FROM --platform=linux/amd64 python:3.9-slim-buster

LABEL IMAGE="ROIQUANT"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="True"

RUN apt-get update
RUN apt-get -y install libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN mkdir /src

COPY files/roi_quantifier.py /src
COPY files/reportdriver_roi_quantifier.py /src

RUN mkdir /refs
COPY refs/ /refs

CMD ["python3","-u","src/reportdriver_roi_quantifier.py"]
