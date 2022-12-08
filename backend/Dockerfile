# FROM nvidia/cuda:11.7.0-base-ubuntu20.04
FROM pytorch/pytorch:latest

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# RUN apt update
# RUN apt install software-properties-common
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update
# RUN apt-get install -y python3.9
# RUN apt install -y python3-pip
# RUN apt-get -y install python3-pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# RUN pip install torch torchvision torchaudio

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8084"]