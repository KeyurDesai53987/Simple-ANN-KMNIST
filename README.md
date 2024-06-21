# Simple-ANN-KMNIST
## Introduction
- Code of Simple Artificial Neural Network Model for KMNIST Datasets
## Requirements
- **Python 3.11**
install PyTorch-CUDA using the below line on your virtual environment (Make sure you have already installed cuda on your device)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
### Python Packages
- **numpy**
- **pandas**
- **matplotlib**
- **scikit-learn**
- **torch**
- **torchvision**
## To RUN .ipynb
- Open final.ipynb (Jupyter Notebook File) and run all the cells on Jupyter lab or notebook 
## To RUN .py
- Open cmd and goto the directory where all the files are saved
- Run below command
```
python3 main.py
```
## Results
- You can find the results of code in the same directory named 'Results.csv' and 'Test Results.csv'
- we got Test Accuracy as below

| -- | Optimizer | Learning Rate | Batch Size | Test Accuracy |
| -- | -- | -- | -- | -- |
| 0 | adam | 0.0001 | 32 | 0.9179 |
| 1 | rmsprop | 0.0010 | 64 | 0.9167 |
| 2 | adamw | 0.0010 | 128 | 0.9134 |
