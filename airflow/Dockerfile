FROM apache/airflow:2.8.1

USER root
COPY requirements.txt /tmp/requirements.txt
RUN chmod 644 /tmp/requirements.txt

USER airflow
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

CMD ["airflow", "webserver"]











