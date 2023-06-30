FROM --platform=linux/amd64 python:3.9-slim-buster

LABEL IMAGE="HTML2PDF"
LABEL VERSION="0.1.0"
LABEL CI_IGNORE="True"

RUN apt-get update
RUN apt-get -y install libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN mkdir /src

COPY files/html2pdf.py /src
COPY files/reportdriver_html2pdf.py /src

RUN mkdir /logos
COPY logos/ /logos

CMD ["python3","-u","src/reportdriver_html2pdf.py"]
