FROM python:3.10.12-alpine3.18

LABEL IMAGE="AWS-S3-BUCKET-CLEANER"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="True"

RUN pip install --upgrade pip

COPY files/ /kaapana/app/

RUN python -m pip install --no-cache-dir -r /kaapana/app/requirements.txt

CMD ["python3","-u","/kaapana/app/start.py"]
