import torch


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out[:, -1])
        return predictions
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        Q, K, V = self.queries(x), self.keys(x), self.values(x)

        Q = Q.view(Q.shape[0], -1, self.heads, self.head_dim)
        K = K.view(K.shape[0], -1, self.heads, self.head_dim)
        V = V.view(V.shape[0], -1, self.heads, self.head_dim)

        energy = Q @ K.permute(0, 1, 3, 2)
        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = attention @ V

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.shape[0], -1, self.embed_size)
        return self.fc_out(out)
    
class FeedForwardLayer(torch.nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(FeedForwardLayer, self).__init__()
        self.fc = torch.nn.Linear(embed_size, hidden_size)
        self.fc_out = torch.nn.Linear(hidden_size, embed_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        return self.fc_out(x)
    

class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = FeedForwardLayer(embed_size, 4 * embed_size)
        
        self.lm1 = torch.nn.LayerNorm(embed_size)
        self.lm2 = torch.nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attention(self.lm1(x))
        x = x + self.feed_forward(self.lm2(x))

        return x
    

class Transformer(torch.nn.Module):
    def __init__(self, dim, n_head, n_layers):
        super(Transformer, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(dim, n_head)
                for _ in range(n_layers)
            ]
        )

        self.out = torch.nn.Linear(dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.out(x)