import transformers
import stanza
from allennlp.predictors.predictor import Predictor

transformers.BertTokenizer.from_pretrained('bert-base-uncased')
transformers.BertModel.from_pretrained('bert-base-uncased')
#transformers.BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
#transformers.BertModel.from_pretrained("bert-large-uncased-whole-word-masking")
stanza.download("en")
Predictor.from_path("models/srl_model/bert-base-srl-2019.06.17.tar.gz")
