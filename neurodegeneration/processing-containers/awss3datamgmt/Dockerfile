FROM python:3.10.12-alpine3.18

LABEL IMAGE="AWS-S3-Data-Mgmt"
LABEL VERSION="0.1.3"
LABEL CI_IGNORE="True"

RUN pip install --upgrade pip

COPY files/ /kaapana/app/

RUN mkdir /tempIn
RUN mkdir /tempOut

RUN python -m pip install --no-cache-dir -r /kaapana/app/requirements.txt

CMD ["python3","-u","/kaapana/app/start.py"]
