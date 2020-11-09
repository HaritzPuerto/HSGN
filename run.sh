cd SAE/
echo 'Computing the relevant documents'
python main.py ../input_sample.json
echo 'Done'
cd ../
python run_prediction.py