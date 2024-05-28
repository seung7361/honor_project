import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from model import ARIMA, RNNModel, LSTMModel, Transformer, TimeSeriesDataset


input_dim = 1
hidden_dim = 64
output_dim = 1
num_layers = 2
num_epochs = 10
learning_rate = 1e-3

p = 500
d = 1
q = 500

device = "cuda:7"


df = pd.read_csv("data/mise.csv")
scaler = MinMaxScaler(feature_range=(-1, 1))
df['mise'] = scaler.fit_transform(df['mise'].values.reshape(-1, 1))
df = df['mise'].values

seq_length = 1000
train_size = int(len(df) * 0.8)
train_data = df[train_size - seq_length:train_size]
test_data = df[train_size - seq_length:]

rnn_model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
lstm_model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
transformer_model = Transformer(input_dim, hidden_dim, output_dim, num_layers).to(device)
arima_model = ARIMA(p, d, q).to(device)

rnn_model.load_state_dict(torch.load("models/rnn_model.pth"))
lstm_model.load_state_dict(torch.load("models/lstm_model.pth"))
transformer_model.load_state_dict(torch.load("models/transformer_model.pth"))
arima_model.load_state_dict(torch.load("models/arima_model.pth"))

rnn_model.eval()
lstm_model.eval()
transformer_model.eval()
arima_model.eval()

criterion = nn.MSELoss()

rnn_loss, lstm_loss, transformer_loss, arima_loss = 0, 0, 0, 0
true, rnn_output, lstm_output, transformer_output, arima_output = [], [], [], [], []
input_seq = torch.tensor(train_data[-seq_length:]).to(device, dtype=torch.float32)
rnn_output, lstm_output, transformer_output, arima_output = input_seq.clone(), input_seq.clone(), input_seq.clone(), input_seq.clone()

with torch.no_grad():
    for i in range(len(test_data) - seq_length):
        true_label = torch.tensor(test_data[i]).to(device)

        rnn_result = rnn_model(rnn_output[-seq_length:].unsqueeze(0).to(device))
        lstm_result = lstm_model(lstm_output[-seq_length:].unsqueeze(0).to(device))
        transformer_result = transformer_model(transformer_output[-seq_length:].unsqueeze(0).to(device))
        arima_result = arima_model(arima_output[-seq_length:].unsqueeze(0).to(device))

        
        rnn_loss += criterion(rnn_result, true_label).item()
        lstm_loss += criterion(lstm_result, true_label).item()
        transformer_loss += criterion(transformer_result, true_label).item()
        arima_loss += criterion(arima_result, true_label).item()
        

        rnn_output = torch.cat([rnn_output.cpu(), rnn_result.cpu()])
        lstm_output = torch.cat([lstm_output.cpu(), lstm_result.cpu()])
        transformer_output = torch.cat([transformer_output.cpu(), transformer_result.cpu()])
        arima_output = torch.cat([arima_output.cpu(), arima_result.unsqueeze(0).cpu()])
        true.append(test_data[i])


rnn_loss /= len(true)
lstm_loss /= len(true)
transformer_loss /= len(true)
arima_loss /= len(true)


rnn_output = rnn_output[seq_length:].numpy()
lstm_output = lstm_output[seq_length:].numpy()
transformer_output = transformer_output[seq_length:].numpy()
arima_output = arima_output[seq_length:].numpy()



plt.figure(figsize=(18, 6))
plt.plot(range(len(train_data)), train_data, label='train')
plt.plot(range(len(train_data), len(train_data) + len(true)), true, label='true')
plt.plot(range(len(train_data), len(train_data) + len(rnn_output)), rnn_output, label='rnn')
plt.plot(range(len(train_data), len(train_data) + len(lstm_output)), lstm_output, label='lstm')
plt.plot(range(len(train_data), len(train_data) + len(transformer_output)), transformer_output, label='transformer')
plt.plot(range(len(train_data), len(train_data) + len(arima_output)), arima_output, label='arima')
plt.legend()
plt.savefig("result.png")

print(f"RNN Loss: {rnn_loss / len(true)}")
print(f"LSTM Loss: {lstm_loss / len(true)}")
print(f"Transformer Loss: {transformer_loss / len(true)}")
print(f"ARIMA Loss: {arima_loss}")
