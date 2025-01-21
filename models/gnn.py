import torch
from torch_geometric.nn import TransformerConv



class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        
        self.conv1 = TransformerConv(in_channels, in_channels // 4, heads=4,
                                    dropout=0.5, edge_dim=edge_dim)
        
        self.conv2 = TransformerConv(in_channels, out_channels // 4, heads=4,
                                    dropout=0.5, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        # print(x.size(), edge_index.size(), edge_attr.size())
        x = self.conv1(x, edge_index, edge_attr)
        return self.conv2(x, edge_index, edge_attr)