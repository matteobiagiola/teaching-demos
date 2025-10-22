FROM python:3.11-slim

WORKDIR /home

ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN apt update -y

RUN pip install hypothesis[cli]==6.142.1
RUN pip install pytest==8.4.2

USER $USERNAME

ENTRYPOINT [ "/bin/bash" ]
