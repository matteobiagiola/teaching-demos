FROM python:3.11-slim

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

# dnn-tip uses its numerically stable KDE (needed by LSA) only with scipy < 1.10,
# which in turn requires numpy < 1.26 and a scikit-learn version compatible with both
RUN pip install numpy==1.25.2 scipy==1.9.3 scikit-learn==1.5.2 dnn-tip==0.1.1

USER $USERNAME

ENTRYPOINT [ "/bin/bash" ]
