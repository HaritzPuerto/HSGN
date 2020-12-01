cd SAE/
echo 'Computing the relevant documents'
python main.py ../data/external/input.json
echo 'Done'
cd ../
python run_prediction.py