import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.rnn = nn.GRU(input_size=config.embedding_size,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layers,
                          bidirectional=config.bidirectional,
                          batch_first=True)
        pass

    def forward(self, words, length):
        embed = self.embedding(words)
        inputs = pack_padded_sequence(embed, length, True)
        _, hidden = self.rnn(inputs, self.init_hidden())

        return hidden

    def init_hidden(self):
        layers = self.config.num_layers * 2 if self.config.bidirectional else self.config.num_layers
        return torch.zeros((layers, self.config.batch_size, self.config.hidden_size), dtype=torch.float64,
                           device=self.config.device)

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.GRU(input_size=config.embedding_size,
                          hidden_size=config.hidden_size * 2 if config.bidirectional else config.hidden_size,
                          num_layers=config.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, hidden, target):
        inputs = self.embedding(target[:, :-1])
        out, _ = self.rnn(inputs, hidden)
        outs = self.fc(out)
        return outs
