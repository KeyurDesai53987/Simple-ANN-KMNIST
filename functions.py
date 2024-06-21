import pandas as pd
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold

from simpleANN import SimpleANN

class Functions():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pass

    def load_data(self):
        # Load and normalize the KMNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset

    # Hyperparameter Tuning and Cross-Validation
    def train_evaluate_test(self, train_dataset, optimizer_name, learning_rate, batch_size, num_epochs=10):

        device = self.device

        # K-Fold Cross-Validation
        kf = KFold(n_splits=5)
        val_accuracy_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            print(f"Fold {fold + 1}/{kf.n_splits}")

            # Creating data loaders for cross-validation
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader_cv = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader_cv = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

            model = SimpleANN().to(device)  # Move model to GPU

            if optimizer_name == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
            criterion = nn.MSELoss()

            # Training
            model.train()
            for epoch in range(num_epochs): 
                # print(f"  Epoch {epoch + 1}/num_epochs")
                running_loss = 0.0
                train_correct = 0
                for batch_idx, (inputs, targets) in enumerate(train_loader_cv):
                    inputs, targets = inputs.to(device), targets.to(device)
                    one_hot_targets = F.one_hot(targets, num_classes=10).float()

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, one_hot_targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_correct += (predicted == targets).sum().item()
                    # if batch_idx % 10 == 0:
                    #     print(f"    Batch {batch_idx + 1}/{len(train_loader_cv)}, Loss: {running_loss / (batch_idx + 1):.4f}")

                train_loss = running_loss / len(train_loader_cv)
                train_accuracy = train_correct / len(train_sampler)

                # Validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(val_loader_cv):
                        inputs, targets = inputs.to(device), targets.to(device)
                        one_hot_targets = F.one_hot(targets, num_classes=10).float()
                        outputs = model(inputs)
                        loss = criterion(outputs, one_hot_targets)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == targets).sum().item()
                        # if batch_idx % 10 == 0:
                        #     print(f"    Validation Batch {batch_idx + 1}/{len(val_loader_cv)}")

                val_loss /= len(val_loader_cv)
                val_accuracy = val_correct / len(val_sampler)

                print(f"  Epoch {epoch + 1}/{num_epochs} - train_loss: {train_loss:.4f} - train_acc: {train_accuracy:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f}")

            val_accuracy_scores.append(val_accuracy)
            print(f"  Fold {fold + 1} Accuracy: {val_accuracy:.4f}")

        return model, np.mean(val_accuracy_scores)
    
    # Training the final model on the entire training dataset
    def train_final_model(self, train_dataset, optimizer_name, learning_rate, batch_size, num_epochs=100):

        device = self.device

        model = SimpleANN().to(device)

        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model.train()
        for epoch in range(num_epochs):
            # print(f"Epoch {epoch + 1}/{num_epochs}")
            running_loss = 0.0
            train_correct = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                one_hot_targets = F.one_hot(targets, num_classes=10).float()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, one_hot_targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == targets).sum().item()
                # if batch_idx % 10 == 0:
                #     print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {running_loss / (batch_idx + 1):.4f}")

            train_loss = running_loss / len(train_loader)
            train_accuracy = train_correct / len(train_dataset)
            print(f"Epoch {epoch + 1}/{num_epochs} - train_loss: {train_loss:.4f} - train_acc: {train_accuracy:.4f}")

        return model

    # Evaluate the model on the test set
    def evaluate_test_set(self, model, test_dataset, batch_size):

        device = self.device

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == targets).sum().item()

        test_accuracy = test_correct / len(test_dataset)
        print(f"Test Set Accuracy: {test_accuracy:.4f}")
        return test_accuracy