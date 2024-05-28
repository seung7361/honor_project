import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from model import ARIMA, RNNModel, LSTMModel, Transformer, TimeSeriesDataset

import os
os.environ["CUDA_VISIBLE_DEVICE"] = "7"

# Load and preprocess data
data = pd.read_csv('data/mise.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
data['mise'] = scaler.fit_transform(data['mise'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences

seq_length = 1000
sequences = create_sequences(data['mise'].values, seq_length)

train_size = int(len(sequences) * 0.8)
train_sequences = sequences[:train_size]
test_sequences = sequences[train_size:]

train_dataset = TimeSeriesDataset(train_sequences)
test_dataset = TimeSeriesDataset(test_sequences)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(seq.unsqueeze(-1))
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')



# Training settings
input_dim = 1
hidden_dim = 64
output_dim = 1
num_layers = 2
num_epochs = 10
learning_rate = 1e-3

p = 1000
d = 1
q = 1000

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:7"
print("device:", device)

# Initialize models, loss function, and optimizer
rnn_model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
lstm_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
transformer_model = Transformer(input_dim, hidden_dim, output_dim, num_layers).to(device)
arima_model = ARIMA(p, d, q).to(device)

criterion = nn.MSELoss()

rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate)
arima_optimizer = optim.Adam(arima_model.parameters(), lr=learning_rate)

# Train RNN model
print("Training RNN Model")
train_model(rnn_model, train_loader, criterion, rnn_optimizer, num_epochs)

# Train LSTM model
print("Training LSTM Model")
train_model(lstm_model, train_loader, criterion, lstm_optimizer, num_epochs)

print("Training Transformer Model")
train_model(transformer_model, train_loader, criterion, transformer_optimizer, num_epochs)

print("Training ARIMA Model")
train_model(arima_model, train_loader, criterion, arima_optimizer, num_epochs)

# Save the models
torch.save(rnn_model.state_dict(), 'models/rnn_model.pth')
torch.save(lstm_model.state_dict(), 'models/lstm_model.pth')
torch.save(transformer_model.state_dict(), 'models/transformer_model.pth')
torch.save(arima_model.state_dict(), 'models/arima_model.pth')