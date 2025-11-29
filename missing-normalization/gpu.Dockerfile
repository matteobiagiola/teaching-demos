FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /home

ARG USERNAME
ARG USER_UID
ARG USER_GID

ENV USERNAME=$USERNAME

# Create the user
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -ms /bin/bash $USERNAME

RUN apt update -y

RUN pip install torch==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install matplotlib==3.10.6

USER $USERNAME

ENTRYPOINT [ "/bin/bash" ]
