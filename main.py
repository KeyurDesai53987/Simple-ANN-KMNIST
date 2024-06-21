import pandas as pd
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold

from functions import Functions

func = Functions()

# Set device to GPU if available
device = func.device

# Download and Load Datasets
train_dataset, test_dataset = func.load_data()

results = []

# Define hyperparameters
learning_rates = [1e-3]
batch_sizes = [32]

for optimizer_name in ['adam', 'rmsprop', 'adamw']:
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"Testing {optimizer_name} with learning rate {lr} and batch size {bs}")
            model, val_acc = func.train_evaluate_test(train_dataset, optimizer_name, lr, bs, 1)
            results.append((optimizer_name, lr, bs, val_acc))

results_df = pd.DataFrame(results, columns=['Optimizer', 'Learning Rate', 'Batch Size', 'Accuracy'])
best_results = results_df.loc[results_df.groupby('Optimizer')['Accuracy'].idxmax()]
best_results.reset_index(drop=True, inplace=True)

test_results = []

for loop in range(best_results.shape[0]):
    opt = best_results['Optimizer'][loop]
    lr = best_results['Learning Rate'][loop]  
    bs = int(best_results['Batch Size'][loop])

    # Train final model
    final_model = func.train_final_model(train_dataset, opt, lr, bs, 1)
    test_accuracy = func.evaluate_test_set(final_model, test_dataset, bs)
    test_results.append((opt, lr, bs, test_accuracy))

test_results_df = pd.DataFrame(results, columns=['Optimizer', 'Learning Rate', 'Batch Size', 'Test Accuracy'])

results_df.to_csv('Results.csv', index=False)
test_results_df.to_csv('Test Results.csv', index=False)