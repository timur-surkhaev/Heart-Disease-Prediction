# The Capstone Project 1 for ML-Zoomcamp 2024

Predicting presence or absence of heart disease based on a patient's health and demographic data

Data is from https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

EDA, model training, and selection are provided in notebooks/diabetes_prediction.ipynb.

The final process of model training, selection, and saving is provided in source/train.py.

<b>NB!</b> Only the libraries necessary for automatic training and testing of models, as well as for launching the prediction service, are included in the virtual environment. The .ipynb notebook is not required for this purpose, so it can be opened in any environment that contains the necessary libraries (seaborn, matplotlib, etc.).


## Containerization

Docker image: https://hub.docker.com/repository/docker/tsurkhaev/heart_disease_prediction/general

1. Pull the docker image: 
```bash
sudo docker pull tsurkhaev/heart_disease_prediction:v0.1
```

2. Run the container: 
```bash
sudo docker run --rm -it -p 0.0.0.0:8000:8000 tsurkhaev/heart_disease_prediction:v1.0
```

Optionally. Build the container locally (if you have a local copy of the repository; see below): 
```bash
sudo docker build -t heart_disease_prediction:local .
```

## Virtual environment

Installing the virtual environment:

1. Get the repository:
```bash
cd ~/local_path_to_the_project
git clone https://github.com/timur-surkhaev/Heart-Disease-Prediction.git
```

2. Install the virtual env:
```bash
pip install pipenv
pipenv install --ignore-pipfile
```

3. Activate the virtual environment (in the root project directory):
```bash
pipenv shell
```

## Test

After the Docker container has been launched, you can send a test request in the activated virtual environment
```bash
python3 tests/test_request_01.py
```
