FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /home

ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN apt update -y
RUN pip install matplotlib==3.10.6

USER $USERNAME

ENTRYPOINT [ "/bin/bash" ]
