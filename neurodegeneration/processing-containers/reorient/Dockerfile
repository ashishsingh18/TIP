FROM --platform=linux/amd64 python:3.9-slim-buster

LABEL IMAGE="REORIENT"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="True"


COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN mkdir /src

COPY files/reorient.py /src
COPY files/reportdriver_reorient.py /src

CMD ["python3","-u","src/reportdriver_reorient.py"]
