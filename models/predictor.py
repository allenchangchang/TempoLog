import torch
from torch.nn import Linear


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)
        # self.lin_final = nn.Sequential(
        #     nn.Linear(in_channels, in_channels),
        #     nn.ReLU(),
        #     nn.Linear(in_channels, 1)
        # )

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)
    

class LogTGNPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, out_channels)
        # self.lin_final = nn.Sequential(
        #     nn.Linear(in_channels, in_channels),
        #     nn.ReLU(),
        #     nn.Linear(in_channels, 1)
        # )

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)