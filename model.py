import torch


class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()

        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.scale_factor = self.head_dim ** -0.5

        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)
        self.out = torch.nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.query(x).reshape(b, n, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(b, n, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(b, n, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale_factor
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.dim)
        return self.out(out)
    

class FeedForwardLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 4 * dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()

        self.layernorm1 = torch.nn.LayerNorm(dim)
        self.layernorm2 = torch.nn.LayerNorm(dim)

        self.attention = MultiHeadAttention(dim, n_head)
        self.ff = FeedForwardLayer(dim)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.ff(self.layernorm2(x))
        return x
    

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(hidden_dim, 8) for _ in range(num_layers)]
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc(x)[:, -1, :]
        return x


class ARIMA(torch.nn.Module):
    def __init__(self, p, d, q):
        super(ARIMA, self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.ar = torch.nn.Linear(p, 1)
        self.ma = torch.nn.Linear(q, 1)
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        x = x.reshape(-1)
        for _ in range(self.d):
            x = x[1:] - x[:-1]

        ar = self.ar(x[-self.p:])
        ma = self.ma(x[-self.q:])

        return self.fc(ar + ma)
    

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences[idx]
        return torch.FloatTensor(seq), torch.FloatTensor([label])