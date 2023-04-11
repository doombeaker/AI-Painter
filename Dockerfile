FROM python:3.8.16-alpine3.16

RUN mkdir /app
ADD . /app

WORKDIR /app


RUN python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
