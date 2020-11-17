from torch import nn


class BottleneckBase(nn.Module):
    def __init__(self, encoder, decoder, compressor=None, decompressor=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.compressor = compressor
        self.decompressor = decompressor

    def forward(self, x):
        z = self.encoder(x)
        if self.compressor is not None:
            z = self.compressor(z)

        if self.decompressor is not None:
            z = self.decompressor(z)
        return self.decoder(z)
