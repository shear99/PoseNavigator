import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DynamicGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, **kwargs):
        super(DynamicGATConv, self).__init__(in_channels, out_channels, heads=heads, concat=concat, **kwargs)

    def forward(self, x, edge_index, edge_weight=None):
        # Edge Weight를 Attention Mechanism에 포함시킵니다.
        return super(DynamicGATConv, self).forward(x, edge_index, edge_attr=edge_weight)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = DynamicGATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = DynamicGATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x
