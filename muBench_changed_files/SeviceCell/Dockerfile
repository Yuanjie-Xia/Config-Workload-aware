FROM loongspeed/microservice_v4-screen:latest
COPY CellController-mp.py /app/CellController-mp.py
COPY ExternalServiceExecutor.py /app/ExternalServiceExecutor.py
COPY InternalServiceExecutor.py /app/InternalServiceExecutor.py
CMD ["/bin/bash", "/app/start-mp.sh"]