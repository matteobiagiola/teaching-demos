FROM python:3.11-slim

WORKDIR /home

ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN apt update -y

RUN pip install statsmodels==0.14.0
RUN pip install matplotlib==3.7.1
RUN pip install numpy==1.25.2

USER $USERNAME

ENTRYPOINT [ "/bin/bash" ]
