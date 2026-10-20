"""
Neural-network architectures (PyTorch). Every model maps a window  x: (batch, seq_len, n_features)
to one number per sample: the predicted utilisation of the next 15 minutes.

  mlp          - flatten the window -> 2 hidden layers            (simplest deep-learning baseline)
  lstm / gru   - recurrent encoder, last hidden state -> head      (classic sequence models)
  transformer  - self-attention encoder over the 16 time steps     (the "transformer" model)
"""
import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, n_features: int, seq_len: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        d = n_features * seq_len
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze(-1)


class RNNRegressor(nn.Module):
    def __init__(self, n_features: int, rnn_type: str = "lstm", hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(n_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)                 # (batch, seq_len, hidden)
        return self.head(out[:, -1]).squeeze(-1)


class TrafficTransformer(nn.Module):
    """
    Encoder-only Transformer for time series.
      1. Linear layer lifts the n_features of each 15-min step to d_model dimensions
      2. a learned positional embedding tells the model which step is which (oldest .. newest)
      3. N x (multi-head self-attention + feed-forward) encoder layers
      4. last step + average over all steps -> small head -> next-15-min utilisation
    No causal mask is needed: every input step is in the past relative to the prediction.
    """

    def __init__(self, n_features: int, seq_len: int, d_model: int = 64, nhead: int = 4,
                 layers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by the number of heads"
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.input_proj(x) + self.pos_emb          # (batch, seq_len, d_model)
        h = self.norm(self.encoder(h))
        pooled = torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1)


MODEL_NAMES = ["mlp", "lstm", "gru", "transformer"]


def build_model(name: str, n_features: int, seq_len: int, hidden: int = 64, d_model: int = 64,
                layers: int = 2, heads: int = 4, dropout: float = 0.1) -> nn.Module:
    """Create a model by name. `hidden` is the RNN width (MLP uses 4x that), `d_model` the transformer width."""
    if name == "mlp":
        return MLPRegressor(n_features, seq_len, hidden=hidden * 4, dropout=dropout)
    if name in ("lstm", "gru"):
        return RNNRegressor(n_features, rnn_type=name, hidden=hidden, layers=layers, dropout=dropout)
    if name == "transformer":
        return TrafficTransformer(n_features, seq_len, d_model=d_model, nhead=heads, layers=layers,
                                  dim_ff=d_model * 2, dropout=dropout)
    raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
