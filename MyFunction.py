import torch
import torch.nn as nn
Tensor = torch.Tensor

def positional_encoding(X, num_features, dropout_p=0.1, max_len=48) -> Tensor:
    r'''
    Params：
        - num_features: input dimensions
        - dropout_p: the probabilities of prodropout
        - max_len: default 512

    Shape：
        - input: [batch_size, seq_length, num_features]
        - output: [batch_size, seq_length, num_features]

    Example：
        - X = torch.randn((2,4,10))
        - X = positional_encoding(X, 10)
        - print(X.shape)
        - torch.Size([2, 4, 10])
    '''

    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1, max_len, num_features))
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)
    P[:, :, 0::2] = torch.sin(X_)
    P[:, :, 0::2] = torch.cos(X_)
    X = X + P[:, :X.shape[1], :].to(X.device)
    return dropout(X)
