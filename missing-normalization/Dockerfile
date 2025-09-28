FROM python:3.11-slim

WORKDIR /home

RUN apt update -y

RUN pip install torch==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install matplotlib==3.10.6

ENTRYPOINT [ "/bin/bash" ]
