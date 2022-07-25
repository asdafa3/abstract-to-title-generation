import torch
import subprocess
import os

from zmq import device

def mount_root():
    subprocess.run(["pip", "install", "git_root"])

    PROJECT_ROOT = None
    in_colab = 'google.colab' in str(get_ipython())

    if in_colab:
      print('Running on CoLab')
      PROJECT_ROOT = "/content/drive/MyDrive/DL4NLP/abstract-to-title-generation"
      from google.colab import drive
      drive.mount('/content/drive')

    else:
      print('Running on local machine')
      from git_root import git_root
      PROJECT_ROOT = git_root()
    return PROJECT_ROOT

def get_device_string():
    if torch.cuda.is_available():
        device_id = "cuda:0"
    elif torch.backends.mps.is_available():
        device_id = "mps"
    else:
        device_id = "cpu"

    print(device_id)

    return device_id
    

def get_torch_device():
    device_id = get_device_string()
    return torch.device(device_id)

PROJECT_ROOT = mount_root()
DEVICE_ID = get_device_string()

os.chdir(PROJECT_ROOT)

DATA_DIR = f"{PROJECT_ROOT}/data"
OUTPUT_DIR = f"{PROJECT_ROOT}/output"

MODEL_DIR = f"{PROJECT_ROOT}/model"
RL = f"{PROJECT_ROOT}/RL_bart_xsum"

DATASET_140_ANNOTATED_JSON = f'{OUTPUT_DIR}/annotated_json/dataset_140samples.json'

FILTERED_DATA = f"{DATA_DIR}/filtered"
