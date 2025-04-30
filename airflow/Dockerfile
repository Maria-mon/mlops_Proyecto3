FROM apache/airflow:2.8.1

USER root
COPY requirements.txt /tmp/requirements.txt
RUN chmod 644 /tmp/requirements.txt

USER airflow
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

RUN mkdir -p /opt/airflow/logs/scheduler \
             /opt/airflow/logs/webserver \
             /opt/airflow/logs/task \
             /opt/airflow/logs/dag_processor_manager \
    && chown -R airflow:root /opt/airflow/logs

CMD ["airflow", "webserver"]












