#FROM tensorflow/tensorflow:2.1.0-gpu-py3
FROM tensorflow/tensorflow:1.15.2-gpu-py3
ENV http_proxy http://proxy.esl.cisco.com:80/
ENV https_proxy http://proxy.esl.cisco.com:80/
ENV DEBIAN_FRONTEND noninteractive
MAINTAINER Rob Liston <rliston@cisco.com>

RUN apt-get update
RUN apt-get install -y --fix-missing tk-dev python3-tk-dbg python3-cairo-dev python3-cairocffi libgtk-3-dev python3-pyqt5
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y apt-utils build-essential software-properties-common cmake git yasm pkg-config wget expect
RUN apt-get install -y vim eog bc mplayer sudo curl zip tcpdump tcptrace wireshark-gtk tshark openssh-client

RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN pip3 install matplotlib
RUN pip3 install opencv-python

RUN apt-get update -y && apt-get install -y python-requests gnuradio gnuradio-dev
RUN apt-get update -y && apt-get install -y python3-requests
RUN /usr/lib/uhd/utils/uhd_images_downloader.py

RUN pip3 install scapy
RUN pip3 install hexhamming
RUN pip install python-lifelib
RUN python -c 'import lifelib ; print('lifelib',lifelib.__version__) ; lifelib.load_rules("b3s23")'

RUN ldconfig
ENTRYPOINT ["/bin/bash", "-i"]
