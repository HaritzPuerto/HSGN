from pytorch/pytorch
#RUN git clone https://github.com/HaritzPuerto/HSGN.git /workspace/ml-workspace/HSGN
WORKDIR /HSGN/
RUN mkdir data
RUN mkdir data/external

RUN apt-get update && apt install build-essential -y --no-install-recommends
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O data/external/input.json

COPY . .
#CMD ["python", "run_prediction.py"]
#CMD ["python", "src/utils/hotpot_evaluate_v1.py", "./predictions.json", "data/external/input.json"]