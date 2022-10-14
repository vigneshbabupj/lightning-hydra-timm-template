FROM python:3.10-slim-buster

ENV GRADIO_SERVER_PORT 80

WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

EXPOSE 80

ENTRYPOINT [ "python3", "s3_cifar10_docker.py" ]