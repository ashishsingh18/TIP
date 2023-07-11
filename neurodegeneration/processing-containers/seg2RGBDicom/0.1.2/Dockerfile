FROM python:3.9-slim

LABEL IMAGE="seg2RGBDicom"
LABEL VERSION="0.1.2"
LABEL CI_IGNORE="True"

RUN pip install --upgrade pip

RUN pip3 install -r requirements.txt

COPY files/start.py /

CMD ["python3","-u","/start.py"]