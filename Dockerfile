FROM python:3.8-slim-buster
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY ./main.py /app/
CMD ["python", "-u", "/app/main.py"]