import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Load the dataset
file_path = 'data/mise_2_weekly.csv'
data = pd.read_csv(file_path)

# Preprocess the data
values = data['mise'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# Convert the data into sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 20
X, y = create_sequences(scaled_data, seq_length)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Create DataLoader
batch_size = 512
train_data = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Check if GPU is available and move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)

# Plot the true labels and predicted labels
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(train_data)), scaler.inverse_transform(train_data[:][1]), label='Train Data')
# plt.plot(range(len(train_data), len(train_data) + len(y_test)), y_test, label='Test Data')
# plt.plot(range(len(train_data), len(train_data) + len(test_predictions)), test_predictions, label='Predictions')
# plt.legend()
# plt.savefig("predictions.png")


def train(model, device, epochs, lr, train_dataloader, X_test, y_test):
    model.train()

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training the model for {epochs} epochs with learning rate {lr}")

    for epoch in range(epochs):
        for i, (seq, labels) in enumerate(train_dataloader):
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, seq.size(0), model.hidden_layer_size).to(device),
                                 torch.zeros(1, seq.size(0), model.hidden_layer_size).to(device))

            y_pred = model(seq)

            loss = loss_function(y_pred, labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch} train loss: {loss.item()}')


    model.eval()
    test_predictions = []
    seq = X_test[0].unsqueeze(0).to(device)

    for _ in range(len(X_test)):
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                 torch.zeros(1, 1, model.hidden_layer_size).to(device))
            y_pred = model(seq)
            test_predictions.append(y_pred.item())

            seq = torch.cat((seq[:, 1:, :], y_pred.reshape(1, 1, 1)), axis=1)


    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse_loss = nn.MSELoss()
    loss = mse_loss(torch.tensor(test_predictions), torch.tensor(y_test))
    print(f'Mean Squared Error: {loss.item()}')

    return model, test_predictions, y_test


lrs = np.arange(5e-5, 1e-2, 5e-5)[::-1]
lrs = list(lrs)
print(lrs, len(lrs))
epochs_set = list(range(100, 2000, 100))
print(epochs_set, len(epochs_set))
result = []

for epochs in epochs_set:
    out = []
    for lr in lrs:
        model = LSTMModel().to(device)
        model, test_predictions, y_out = train(model, device, epochs, lr, train_dataloader, X_test, y_test)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(train_data)), scaler.inverse_transform(train_data[:][1]), label='Train Data')
        plt.plot(range(len(train_data), len(train_data) + len(y_out)), y_out, label='Test Data')
        plt.plot(range(len(train_data), len(train_data) + len(test_predictions)), test_predictions, label='Predictions')
        plt.legend()
        plt.savefig(f"images/predictions_{epochs}_{lr}.png")
        plt.close()

        loss_fn = nn.MSELoss()
        loss = loss_fn(torch.tensor(test_predictions), torch.tensor(y_out))

        out.append(loss)

    result.append(out)

# Plot the results
plt.figure(figsize=(10, 6))
sns.heatmap(result, annot=True, xticklabels=lrs, yticklabels=epochs_set)
plt.xlabel('Learning Rate')
plt.ylabel('Epochs')
plt.title('Mean Squared Error')

plt.savefig("results.png")