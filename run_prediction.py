from src.data.preprocess_dataset import create_dataloader

data_path = '/workspace/ml-workspace/thesis_git/HSGN/data/'
output = create_dataloader(data_path)

# model_path = ''
# output = create_graphs(data_path)
# model = load_model(model_path)
# pred = model.predict(data_path)
