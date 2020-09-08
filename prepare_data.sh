mkdir data
mkdir data/external
mkdir data/processed
mkdir data/processed/training
mkdir data/processed/dev
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O data/external/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O data/external/hotpot_dev_distractor_v1.json