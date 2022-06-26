import torch
import subprocess
import os

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

def get_device():
    if torch.cuda.is_available():
        device_id = "cuda:0"
    elif torch.backends.mps.is_available():
        device_id = "mps"
    else:
        device_id = "cpu"

    print(device_id)

    return torch.device(device_id)

PROJECT_ROOT = mount_root()

os.chdir(PROJECT_ROOT)

DATA_DIR = f"{PROJECT_ROOT}/data"
OUTPUT_DIR = f"{PROJECT_ROOT}/output"

MODEL_DIR = f"{PROJECT_ROOT}/output"

DATASET_140_ANNOTATED_JSON = f'{OUTPUT_DIR}/annotated_json/dataset_140samples.json'

FILTERED_DATA = f"{DATA_DIR}/filtered"
