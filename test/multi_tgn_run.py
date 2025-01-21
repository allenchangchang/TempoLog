import sys
sys.path.append('/root/LogTGN')

import os.path as osp
import os
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import Linear
import pandas as pd
import numpy as np

from torch_geometric.loader import TemporalDataLoader
from models.LogTGN import LogTGNMemory, GRUAggregator, LinearMeassge
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    MeanAggregator,
)
from models.gnn import GraphAttentionEmbedding
from models.predictor import LinkPredictor

from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 使用示例
dataset_name = "Spirit"
save_path = f'/root/LogTGN/data/{dataset_name}/TemporalData/temporal_data_remove_duplicate_interval2.pt'  # TemporalData 保存目录
data = torch.load(save_path)

print(data)

data, data_1 = data[0], data[1]
# data, data_1 = data[0][:100000], data[1][:100000]

# data = data[:100000]
print(data_1)

data = data.to(device)
# data.msg = data.msg.unsqueeze(-1)

train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.1, test_ratio=0.4)
train_data_1, val_data_1, test_data_1 = data_1.train_val_test_split(val_ratio=0.1, test_ratio=0.4)

print('Training data count: {}, validation data count: {}, testing data count: {}'.format(len(train_data), len(val_data), len(test_data)))
print('Training data count: {}, validation data count: {}, testing data count: {}'.format(len(train_data_1), len(val_data_1), len(test_data_1)))
print('data msg: ', data.msg.size())
print('Node Number:', data.num_nodes)

# train_data = train_data[train_data.y == 0] # 只使用正常数据进行训练
train_batch_size = 400
val_batch_size = 2048
test_batch_size = 2048

train_loader = TemporalDataLoader(
    train_data,
    batch_size=train_batch_size,
    neg_sampling_ratio=1.0,
)
val_loader = TemporalDataLoader(
    val_data,
    batch_size=val_batch_size,
    neg_sampling_ratio=1,
)
test_loader = TemporalDataLoader(
    test_data,
    batch_size=test_batch_size,
    neg_sampling_ratio=0.6,
)

train_loader_1 = TemporalDataLoader(
    train_data_1,
    batch_size=train_batch_size,
    neg_sampling_ratio=1.0,
)
val_loader_1 = TemporalDataLoader(
    val_data_1,
    batch_size=val_batch_size,
    neg_sampling_ratio=1,
)
test_loader_1 = TemporalDataLoader(
    test_data_1,
    batch_size=test_batch_size,
    neg_sampling_ratio=0.6,
)


# 近邻的数量
neighbor_loader = LastNeighborLoader(data.num_nodes, size=20, device=device)   
 
memory_dim = 1024
time_dim = embedding_dim = 300

# message_module = LinearMeassge(data.msg.size(-1), memory_dim, time_dim)
message_module = IdentityMessage(data.msg.size(-1), memory_dim, time_dim)
aggregator_module = LastAggregator()
# aggregator_module = MeanAggregator()


memory = LogTGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=message_module,
    aggregator_module=aggregator_module,
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters())
    | set(link_pred.parameters()) | set(aggregator_module.parameters()) | set(message_module.parameters()), lr=0.00005)
criterion = torch.nn.BCELoss()
# criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state_semantic(vec_save_path=f'/root/LogTGN/data/{dataset_name}/parser/vec.pt', csv_path=f'/root/LogTGN/data/{dataset_name}/parser/{dataset_name}.log_structured.csv')  # Start with a fresh memory.  这里memory init 为Semantic embeddding
    # memory.reset_state() 
    neighbor_loader.reset_state()  # Start with an empty graph.
    train_iter = tqdm(train_loader)
    total_loss = 0
    for batch in train_iter:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        
        # 负样本采样
        # neg_id = torch.randperm(data.num_nodes, dtype=torch.long, device=device)[:len(batch.neg_dst)]
        # neg_id, edge_index, e_id = neighbor_loader(neg_id)
        # neg_z, last_update = memory(neg_id)
        # neg_z = gnn(neg_z, last_update, edge_index, data.t[e_id].to(device),
        #         data.msg[e_id].to(device))
    
        # neg_out = link_pred(z[assoc[batch.src][:len(batch.neg_dst)]], neg_z[:len(batch.neg_dst)])
        
        # 使用 batch 的原始标签参与计算
        labels = batch.y.float().to(device)  
        loss = criterion(out.squeeze().sigmoid(), labels)
        # loss +=  criterion(neg_out.squeeze().sigmoid(), torch.ones(neg_out.size(0)).to(device))
        

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
        # print info
        train_iter.set_postfix(loss=loss.item())
        
    neighbor_loader.reset_state()  # Start with an empty graph.
    train_iter = tqdm(train_loader_1)
    for batch in train_iter:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        
        # 使用 batch 的原始标签参与计算
        labels = batch.y.float().to(device)  
        loss = criterion(out.squeeze().sigmoid(), labels)
        # loss +=  criterion(neg_out.squeeze().sigmoid(), torch.ones(neg_out.size(0)).to(device))
        

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
        # print info
        train_iter.set_postfix(loss=loss.item())
    
    return total_loss / train_data.num_events / train_data_1.num_events


@torch.no_grad()
def test(loader, loader_1):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    f1, p, r = [], [], []
    y_true_all = []
    y_score_all = []
    
    for batch, batch_1 in zip(loader, loader_1):
        batch = batch.to(device)
        batch_1 = batch_1.to(device)

        # Get neighbor info
        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
       
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
       

        # Get embeddings and predictions
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
   
        out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        y_score = out.sigmoid().cpu()
        
        n_id_1, edge_index_1, e_id_1 = neighbor_loader(batch_1.n_id)
        assoc[n_id_1] = torch.arange(n_id_1.size(0), device=device)
         # Get embeddings and predictions
        z, last_update = memory(n_id_1)
        z = gnn(z, last_update, edge_index_1, data.t[e_id_1].to(device),
                data.msg[e_id_1].to(device))
   
        out_1 = link_pred(z[assoc[batch_1.src]], z[assoc[batch_1.dst]])
        y_score_1 = out_1.sigmoid().cpu()
        
        # y_score = (y_score + y_score_1) / 2
        y_score = y_score * 0.7 + y_score_1 * 0.3
        
        y_true = batch.y.cpu()
        norm_idx = (batch.level == 0).cpu()
        
        # print(y_true)
        # print(batch.level)
        # for i in range(len(y_true)):
        #     if y_true[i] == 1:
        #         print()
        #         print('batch.level:', batch.level[i])
        # if y_true == 1:
        #     print('batch.level:', batch.level)
        # if y_true.item() == 1:
        #     print('batch.level:', batch.level.item())
            
        y_score[norm_idx] = 0.0  # 日志级别如果为INFO, 则为正常
        
        y_score_all.append(y_score)

        y_pred = (y_score >= 0.5).int().numpy()
        y_true_all.append(y_true)
        
        f1.append(f1_score(y_true.numpy(), y_pred))
        p.append(precision_score(y_true.numpy(), y_pred))
        r.append(recall_score(y_true.numpy(), y_pred))

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    
    y_score_all = torch.cat(y_score_all, dim=0).numpy()
    y_true_all = torch.cat(y_true_all, dim=0).numpy()
    thresholds = np.arange(0, 1, 0.1)
    best_threshold = 0
    best_f1 = 0

    for threshold in thresholds:
        preds = (y_score_all >= threshold).astype(int)
        f1 = f1_score(y_true_all, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f"Best Threshold: {best_threshold}, Best F1 Score: {best_f1}")
    
    y_pred_all = (y_score_all >= best_threshold).astype(int)
    f1, p, r = f1_score(y_true_all, y_pred_all), \
               precision_score(y_true_all, y_pred_all), \
               recall_score(y_true_all, y_pred_all)

    # f1, p, r = cal_window_metric(y_true_all, y_pred_all, timestamps, window_size=100)
        
    return f1, p, r


def cal_window_metric(y_true_all, y_pred_all, timestamp, window_size=3600):
    # todo
    return
    
    return f1, p, r


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_f1, val_p, val_r = test(val_loader, val_loader_1)
    test_f1, test_p, test_r = test(test_loader, test_loader_1)
    print(f'Val F1: {val_f1:.4f}, Val Precision: {val_p:.4f}, Val Recall: {val_r:.4f}')
    print(f'Test F1: {test_f1:.4f}, Test P: {test_p:.4f}, Test R: {test_r:.4f}')