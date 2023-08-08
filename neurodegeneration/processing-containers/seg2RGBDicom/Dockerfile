FROM python:3.9-slim

LABEL IMAGE="seg2RGBDicom"
LABEL VERSION="0.1.2"
LABEL CI_IGNORE="True"

RUN pip install --upgrade pip

COPY files/ /kaapana/app/

RUN python -m pip install --no-cache-dir -r /kaapana/app/requirements.txt

CMD ["python3","-u","/kaapana/app/start.py"]