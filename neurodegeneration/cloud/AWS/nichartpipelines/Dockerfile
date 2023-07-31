FROM aidinisg/nichartpipelines:0.1.3

RUN pip install boto3

COPY ./mappings /mappings
COPY ./model /models
COPY ./start.py /start.py

CMD ["python3","-u","/start.py"]

