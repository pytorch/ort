FROM ubuntu:18.04

RUN apt-get update

#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN apt-get install -y python3-pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN pip install --upgrade pip

RUN pip install sphinx

RUN pip install --pre onnxruntime-training -f https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_cu111.html
RUN pip install torch-ort

