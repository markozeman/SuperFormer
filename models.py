import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear, Dropout, LayerNorm, MultiheadAttention
from torch.nn import functional as F


class Transformer(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, num_classes):
        super(Transformer, self).__init__()

        transformer_encoder_layer = TransformerEncoderLayer(input_size, num_heads, dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_size * 256, input_size),   # 8192 -> 32
            nn.ReLU(),
            nn.Linear(input_size, num_classes),  # 32 -> 2
        )

        # self.init_weights()

    # def init_weights(self):
    #     initrange = 1e-10
    #     self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, key_padding_mask):
        x = self.transformer_encoder(input, src_key_padding_mask=key_padding_mask)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        output = self.mlp(x)   # feed through MLP
        return output


class MyTransformer(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, num_classes):
        super(MyTransformer, self).__init__()

        transformer_encoder_layer = MyTransformerEncoderLayer(input_size, num_heads, dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_size * 256, input_size),   # 8192 -> 32
            nn.ReLU(),
            nn.Linear(input_size, num_classes),  # 32 -> 2
        )

    def forward(self, input, key_padding_mask):
        x = self.transformer_encoder(input, mask=None, src_key_padding_mask=key_padding_mask)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        output = self.mlp(x)   # feed through MLP
        return output


class MyTransformerEncoderLayer(nn.Module):

    def __init__(self, input_size, num_heads, dim_feedforward, batch_first, dropout=0.0, activation=F.relu):
        super(MyTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=batch_first)

        # implementation of feed-forward model
        self.linear1 = Linear(input_size, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, input_size)

        # # elementwise_affine=False means no trainable parameters (if True, there are 64 trainable parameters)
        # self.norm1 = LayerNorm(input_size, elementwise_affine=False)
        # self.norm2 = LayerNorm(input_size, elementwise_affine=False)

        self.layer_norm = LayerNorm(input_size, elementwise_affine=False)

        self.activation = activation

    def self_attention_block(self, input, key_padding_mask):
        x = self.self_attn(input, input, input, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def ff_block(self, input):
        x = self.linear2(self.dropout(self.activation(self.linear1(input))))
        return self.dropout(x)

    def forward(self, input, src_mask, src_key_padding_mask):
        x = self.layer_norm(input + self.self_attention_block(input, src_key_padding_mask))   # self.norm1
        x = self.layer_norm(x + self.ff_block(x))   # self.norm2
        return x


class MLP(nn.Module):

    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()

        hidden_size = 41    # 41 - to match (or slightly increase) number of parameters in transformer model

        self.mlp = nn.Sequential(
            nn.Linear(input_size * 256, hidden_size),   # 8192 -> 41
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),  # 41 -> 2
        )

    def forward(self, input):
        x = torch.flatten(input, start_dim=1, end_dim=2)
        output = self.mlp(x)
        return output


class AdapterTransformer(nn.Module):

    def __init__(self, input_size, num_heads, num_layers, dim_feedforward, num_classes, bottleneck_size=16):
        super(AdapterTransformer, self).__init__()

        adapter_transformer_encoder_layer = AdapterTransformerEncoderLayer(input_size, num_heads, dim_feedforward,
                                                                           batch_first=True, bottleneck_size=bottleneck_size)
        self.transformer_encoder = TransformerEncoder(adapter_transformer_encoder_layer, num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(input_size * 256, input_size),   # 8192 -> 32
            nn.ReLU(),
            nn.Linear(input_size, num_classes),  # 32 -> 2
        )

    def forward(self, input, key_padding_mask):
        x = self.transformer_encoder(input, mask=None, src_key_padding_mask=key_padding_mask)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        output = self.mlp(x)   # feed through MLP
        return output


class AdapterTransformerEncoderLayer(nn.Module):

    def __init__(self, input_size, num_heads, dim_feedforward, batch_first, dropout=0.0, activation=F.relu, bottleneck_size=16):
        super(AdapterTransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=batch_first)

        # implementation of feed-forward model
        self.linear1 = Linear(input_size, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, input_size)

        # # elementwise_affine=False means no trainable parameters (if True, there are 64 trainable parameters)
        # self.norm1 = LayerNorm(input_size, elementwise_affine=False)
        # self.norm2 = LayerNorm(input_size, elementwise_affine=False)

        self.layer_norm = LayerNorm(input_size, elementwise_affine=False)
        self.layer_norm_trainable = LayerNorm(input_size, elementwise_affine=True)

        self.activation = activation

        self.adapter_layer = AdapterLayer(input_size, bottleneck_size)

    def self_attention_block(self, input, key_padding_mask):
        x = self.self_attn(input, input, input, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def ff_block(self, input):
        x = self.linear2(self.dropout(self.activation(self.linear1(input))))
        return self.dropout(x)

    def forward(self, input, src_mask, src_key_padding_mask):
        x = self.layer_norm(input + self.self_attention_block(input, src_key_padding_mask))  # self.norm1
        x = self.layer_norm(x + self.ff_block(x))  # self.norm2
        x = self.layer_norm_trainable(input + self.adapter_layer(x))    # adapter
        return x


class AdapterLayer(nn.Module):

    def __init__(self, input_size, bottleneck_size=16):
        super(AdapterLayer, self).__init__()

        self.bottleneck_mlp = nn.Sequential(
            nn.Linear(input_size, bottleneck_size),  # 32 -> 16
            nn.ReLU(),
            nn.Linear(bottleneck_size, input_size),  # 16 -> 32
        )

    def forward(self, input):
        x = self.bottleneck_mlp(input)
        output = x + input
        return output

