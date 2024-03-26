FROM python:3.11.8-bookworm

LABEL IMAGE="neuro-sr-generator"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="True"

RUN pip install --upgrade pip

RUN wget -c https://github.com/QIICR/dcmqi/releases/download/v1.3.1/dcmqi-1.3.1-linux.tar.gz -O - | tar -xz

COPY files/ /kaapana/app/

RUN python -m pip install --no-cache-dir -r /kaapana/app/requirements.txt

CMD ["python3","-u","/kaapana/app/start.py"]
