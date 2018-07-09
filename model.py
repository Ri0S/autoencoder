import torch
import torch.nn as nn
import module

class autoEncoder(nn.Module):
    def __init__(self, config):
        super(autoEncoder, self).__init__()
        self.encoder = module.Encoder(config)
        self.decoder = module.Decoder(config)

    def forward(self, data):
        words, target, length = data
        hidden = self.encoder(words, length)
        decoder_out = self.decoder(hidden, target)

        return decoder_out
