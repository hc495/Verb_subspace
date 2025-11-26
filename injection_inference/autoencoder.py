import torch
import torch.nn as nn
import torch.nn.functional as F

class autoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation = "sigmoid",
        bias = "both", # Does help
        residual = False, # Not implemented
        quantized = False, # Not implemented
    ):
        super(autoencoder, self).__init__()
        bias_encoder = True if bias == "encoder" or bias == "both" else False
        bias_decoder = True if bias == "decoder" or bias == "both" else False
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=bias_encoder)
        self.decoder = nn.Linear(hidden_dim, output_dim, bias=bias_decoder)
        if quantized:
            self.encoder.half()
            self.decoder.half()
        self.residual = residual
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "none":
            self.activation = lambda x: x
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(
            self, x,
            return_encoded = False,
            forced_encoded = None,
        ):
        x_new = self.encoder(x)
        x_new = self.activation(x_new)
        if return_encoded:
            encoded = x_new.clone()
        if forced_encoded is not None:
            x_new = forced_encoded
        x_new = self.decoder(x_new)
        if self.residual:
            x_new += x
        if return_encoded:
            return (x_new, encoded)
        else:
            return (x_new)