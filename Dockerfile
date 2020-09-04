#from qipeng/hotpotqa-base:gpu
from pytorch/pytorch
RUN git clone https://github.com/HaritzPuerto/HSGN.git /app
RUN conda create --name myenv --file /app/spec-file.txt
RUN wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O /app/data/external/input.ext
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
CMD ["python", "./run_prediction.py"]