#from qipeng/hotpotqa-base:gpu
from pytorch/pytorch
RUN git clone https://github.com/HaritzPuerto/HSGN.git /app
RUN wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O /app/data/external/input.ext
# Make RUN commands use the new environment:
RUN pip install -r /app/requirements.txt
CMD ["python", "/app/run_prediction.py"]