FROM python:3.10-slim-buster

WORKDIR /opt/src

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

EXPOSE 8080

ENTRYPOINT [ "python3", "demo_cifar10_scripted_docker.py" ]