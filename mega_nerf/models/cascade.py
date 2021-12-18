from torch import nn


class Cascade(nn.Module):
    def __init__(self, coarse: nn.Module, fine: nn.Module):
        super(Cascade, self).__init__()
        self.coarse = coarse
        self.fine = fine
