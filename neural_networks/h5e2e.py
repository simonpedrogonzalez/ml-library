import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import time
from data.datasets import bank_note_dataset


def create_model(input_size, output_size, hidden_size, n_hidden, activation, initialization):
    layers = []
    for _ in range(n_hidden):
        layer = nn.Linear(input_size if len(layers) == 0 else hidden_size, hidden_size)
        if initialization == "xavier":
            nn.init.xavier_uniform_(layer.weight)
        elif initialization == "he":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        layers.append(layer)
        layers.append(activation)
    # Final layer
    final_layer = nn.Linear(hidden_size, output_size)
    nn.init.xavier_uniform_(final_layer.weight) if initialization == "xavier" else None
    layers.append(final_layer)
    layers.append(nn.Sigmoid())  # Ensure outputs are between 0 and 1
    return nn.Sequential(*layers)

def train_model(model, criterion, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0)
    return train_loss / len(train_loader.dataset)

def evaluate_model(model, criterion, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def run_experiment(widths, depths, activations, initialization, train_loader, test_loader, input_size, device):
    results = []
    criterion = nn.BCELoss()
    for activation_name, activation_fn in activations.items():
        for depth in depths:
            for width in widths:
                model = create_model(input_size, 1, width, depth, activation_fn, initialization).to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                for epoch in range(100):
                    train_loss = train_model(model, criterion, optimizer, train_loader, device)
                    test_loss = evaluate_model(model, criterion, test_loader, device)
                    print(f"epoch: {epoch}, activation: {activation_name}, hidden_size: {width}, depth: {depth}, train_loss: {train_loss}, test_loss: {test_loss}")
                    if epoch == 99:
                        results.append({
                            "activation": activation_name,
                            "hidden_size": width,
                            "depth": depth,
                            "train_error": train_loss,
                            "test_error": test_loss
                        })

    return results

def main():
    data = bank_note_dataset().to_numpy()
    data.train_labels = (data.train_labels == 1).astype(int)
    data.test_labels = (data.test_labels == 1).astype(int)


    X_train = torch.tensor(data.train, dtype=torch.float32)
    y_train = torch.tensor(data.train_labels, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(data.test, dtype=torch.float32)
    y_test = torch.tensor(data.test_labels, dtype=torch.float32).unsqueeze(1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    widths = [5, 10, 25, 50, 100]
    depths = [3, 5, 9]
    activations = {"tanh": nn.Tanh(), "relu": nn.ReLU()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for initialization in ["xavier", "he"]:
        results.extend(run_experiment(widths, depths, activations, initialization, train_loader, test_loader, X_train.shape[1], device))

    df = pd.DataFrame(results)
    df.to_csv("neural_networks/reports/results_pytorch.csv", index=False)

    latex_table = df.to_latex(index=False, float_format="%.4f")
    with open("neural_networks/reports/results_pytorch.tex", "w") as f:
        f.write(latex_table)

main()