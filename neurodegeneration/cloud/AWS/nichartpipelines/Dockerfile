FROM aidinisg/nichartpipelines:0.1.4

RUN pip install boto3

COPY ./mappings /mappings
COPY ./model /models
COPY ./start.py /start.py

# ENTRYPOINT python3 /entrypoint.py
CMD ["python3","-u","/start.py"]
