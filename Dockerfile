from pytorch/pytorch
WORKDIR /HSGN/
COPY . .
# RUN mkdir data
# RUN mkdir data/external

RUN apt-get update && apt install build-essential -y --no-install-recommends && apt install -y vim
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
RUN python -c 'import stanza; stanza.download("en")'
RUN python -m spacy download en_core_web_sm
RUN python -c 'import transformers; transformers.BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking"); transformers.BertModel.from_pretrained("bert-large-uncased-whole-word-masking")'
RUN python -c 'from allennlp.predictors.predictor import Predictor; Predictor.from_path("models/srl_model/bert-base-srl-2019.06.17.tar.gz")'
#RUN wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O data/external/input.json

#COPY . .

#CMD python run_prediction.py && python src/utils/hotpot_evaluate_v1.py ./pred.json data/external/input.json