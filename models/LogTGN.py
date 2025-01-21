from torch_geometric.nn import TGNMemory
from torch_geometric.nn.inits import zeros
from typing import Callable
from torch import Tensor
from dataloader.temporaldataloader import get_event_vector, load_event_vectors, load_log_data, load_HDFS_log_data
import copy
from tqdm import tqdm
import torch
from models.gnn import GraphAttentionEmbedding
from models.predictor import LinkPredictor, LogTGNPredictor
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    MeanAggregator,
)

class LogTGNMemory(TGNMemory):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int, time_dim: int, message_module: Callable, aggregator_module: Callable):
        super(LogTGNMemory, self).__init__(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module, aggregator_module)
        self.init_memory = None
        
    def reset_state_semantic(self, vec_save_path='/root/LogTGN/data/BGL/parser/vec.pt', csv_path='/root/LogTGN/data/BGL/parser/BGL.log_structured.csv'):
        """Resets the memory to its initial state."""
        if self.init_memory is None:
            self.init_memory_semantic(vec_save_path=vec_save_path, csv_path=csv_path)
        else:
            self.memory = copy.deepcopy(self.init_memory)
            
        zeros(self.last_update)
        self._reset_message_store()
    
    def init_memory_semantic(self, vec_save_path='/root/LogTGN/data/BGL/parser/vec.pt', csv_path='/root/LogTGN/data/BGL/parser/BGL.log_structured.csv'):
        '''Resets the memory to semantic embedding'''
        vec_save_path = vec_save_path
        csv_path = csv_path
        
        event_vec_df = load_event_vectors(vec_save_path)
        if 'HDFS' in vec_save_path:
            data = load_HDFS_log_data(csv_path)
        else:
            data = load_log_data(csv_path)
        # print(event_vec_df.index)
        
        inited_event_id = []
        node_id_counter = 0
        for i in tqdm(range(len(data)), desc="Reset Memory"):
            if node_id_counter >= self.num_nodes:
                break
            event_id = data.iloc[i]['EventId']
            if event_id in inited_event_id:
                continue
            semantic_embedding = get_event_vector(data.iloc[i]['EventTemplate'], event_vec_df)
            # print(semantic_embedding.size())
            self.memory[node_id_counter] = semantic_embedding   # 语义向量
            inited_event_id.append(event_id)
            node_id_counter += 1
            
        self.init_memory = copy.deepcopy(self.memory)
        
        
class LogTGN(torch.nn.Module):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int, time_dim: int, embedding_dim: int, msg_dim: int, message_module: Callable, aggregator_module: Callable, neighbor_size: int = 10, device=None):
        super(LogTGN, self).__init__()
        self.memory = LogTGNMemory(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module, aggregator_module).to(device)
        
        self.gnn = GraphAttentionEmbedding(memory_dim, embedding_dim, msg_dim, self.memory.time_enc).to(device)
        
        self.device = device
        self.out_channels = 100
        self.predictor = LogTGNPredictor(embedding_dim, self.out_channels).to(device)
        self.neighbor_loader = LastNeighborLoader(num_nodes, size=neighbor_size, device=self.device)
        
        self.assoc = torch.empty(num_nodes, dtype=torch.long).to(device)

        
    def forward(self, x, raw_data):
        '''
            params: x: batch of TemporalDataLoader
                    raw_data: data from raw TemporalData
            output: out: batch of prediction
        '''
        n_id, edge_index, e_id = self.neighbor_loader(x.n_id)
        # print(edge_index, e_id)
        self.assoc[n_id] = torch.arange(n_id.size(0)).to(self.device)

        z, last_update = self.memory(n_id)
        z = self.gnn(z, last_update, edge_index, raw_data.t[e_id],
                raw_data.msg[e_id])
        out = self.predictor(z[self.assoc[x.src]], z[self.assoc[x.dst]])

        # Update memory and neighbor loader with ground-truth state.
        self.memory.update_state(x.src, x.dst, x.t, x.msg)
        self.neighbor_loader.insert(x.src, x.dst)
        self.memory.detach()
        
        return out
    
    def reset_state_semantic(self, vec_save_path='/root/LogTGN/data/BGL/parser/vec.pt', csv_path='/root/LogTGN/data/BGL/parser/BGL.log_structured.csv'):
        """Resets the memory to its initial state."""
        self.memory.reset_state_semantic(vec_save_path='/root/LogTGN/data/BGL/parser/vec.pt', csv_path='/root/LogTGN/data/BGL/parser/BGL.log_structured.csv')
        
    def reset_state(self):
        """Resets the memory to its initial state."""
        self.memory.reset_state()   

        
        
'''
TODO: 1. 实现一个DualLogTGN类，用于对多个LogTGN模型进行加权求和
'''
class DualLogTGN(torch.nn.Module):
    def __init__(self, tgn_models: list[LogTGN]):
        super().__init__()
        self.models = tgn_models
        self.predicter = torch.nn.Linear(len(self.models) * self.models[0].out_channels, 1)
    
    def forward(self, x, total_data):
        """所有的TGN模型都进行一次前向传播，用线形层组合结果"""
        out = []
        for i in range(len(self.models)):
            out.append(self.models[i](x[i], total_data[i]))
        out = torch.cat(out, dim=-1)
        out = self.predicter(out)
        
        return out
    
    def reset_state_semantic(self, vec_save_path='/root/LogTGN/data/BGL/parser/vec.pt', csv_path='/root/LogTGN/data/BGL/parser/BGL.log_structured.csv'):
        """Resets the memory to its initial state."""
        for model in self.models:
            model.reset_state_semantic(vec_save_path='/root/LogTGN/data/BGL/parser/vec.pt', csv_path='/root/LogTGN/data/BGL/parser/BGL.log_structured.csv')
            
    def reset_state(self): 
        """Resets the memory to its initial state."""
        for model in self.models:
            model.reset_state()
        
class GRUAggregator(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # self.gru = torch.nn.GRUCell(input_dim, hidden_dim)
        self.gru = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        hidden_state = msg.new_zeros((dim_size, msg.size(-1)))

        for node_idx in range(dim_size):
            node_mask = (index == node_idx)
            if not node_mask.any():
                continue

            node_msg = msg[node_mask]
            node_t = t[node_mask]
            sorted_indices = torch.argsort(node_t)
            node_msg = node_msg[sorted_indices]

            h = hidden_state[node_idx].unsqueeze(0)
            for m in node_msg:
                h = self.gru(m.unsqueeze(0) + h)
            hidden_state = hidden_state.clone()  # 避免就地修改
            hidden_state[node_idx] = h.squeeze(0)

        return hidden_state
    
class LinearMeassge(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = memory_dim
        in_channels = raw_msg_dim + 2 * memory_dim + time_dim
        self.mlp = torch.nn.Linear(in_channels, memory_dim)

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor,
                t_enc: Tensor):
        input =  torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
        out = self.mlp(input)
        return out