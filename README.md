Thesis
==============================
First of all, you need to run `./prepare_data.sh` to create download HotpotQA and put it in the right folder.

The code contained in this repository is divided into two parts: i) graph creation and ii) model (training and inference)

# Graph Creation
The graphs corresponding to each HotpotQA instance can be created with `./src/data/bottom_up_query_edges.py`. You will need to specify the folder where you want to store the graphs in this script. I used the following folders `./data/processed/training/YOUR_FOLDER` for the graphs of the training set. This scripts assumes the existence of the following files:
* List of named entities in each question (list_ent_query_training.p)
* List of named entities in each paragaph (list_hotpot_ner_no_coref_train.p)
* List of SRL args in each question (dict_ins_query_srl_triples_training.json)
* List of SRL args in each paragraph  (dict_ins_doc_sent_srl_triples.json)

You can create them following `./src/data/preprocess_dataset.py` (you would need to modify a bit the code to add a few lines to store these files) or download them (link at the end of this readme).

# Training the Model

The script `./src/models/GAT_Hierar_Tok_Node_Aggr.py` takes `./data/processed/training/YOUR_FOLDER` (you need to specify it at the begining when loading the graphs) and trains the model. The checkpoints of the model are saved in `./models/YOUR_CHECKPOINT`. This path is defined in the variable `model_path`.

# Inference
First you need some files and pretrained models:
 * Put the input file (HotpotQA dev/test set) on `./data/external/input.json` 
 * Put the pretrained graphQA model on `./models/graph_model`
 * Put the SRL model [link](https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz) on `./models/srl_model`. Do not untar it.
 
Now you can execute `./run.sh`. This will execute the pragraph selection stage of [SAE (Tu et al., 2020)](https://github.com/JD-AI-Research-Silicon-Valley/SAE), then it will create the graphs for those predicted paragraphs, and lastly will inference to obtain the predicted answers. 

# Intermediate Files
- NER and SRL output for the training set: https://drive.google.com/file/d/1R3eyEcwYY5vj2U4X43-WrsPxpotzWS_S/view?usp=sharing

# Precreated Graphs
- Graphs of HotpotQA Train set: https://drive.google.com/file/d/1PiIq7lYpMTmkDddwqabbX5Nn9tnvgZ7S/view?usp=sharing
- Graphs of HotpotQA Dev set: https://drive.google.com/file/d/15mI5HrC6nZf_dMpKujC642spB82HOIaI/view?usp=sharing

# Pretrained Model
- Pretrained model (BERT-based-uncased) https://drive.google.com/file/d/1ZfKgZCnLS3m7BIeT88eKTwnLe_y6rEvn/view?usp=sharing
