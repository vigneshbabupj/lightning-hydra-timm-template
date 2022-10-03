FROM python:3.10-slim-buster

WORKDIR /docker

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

CMD [ "python3", "src/train.py" ]