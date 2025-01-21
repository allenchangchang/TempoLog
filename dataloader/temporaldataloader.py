from sqlite3 import Time
import numpy as np
import sys
sys.path.append('/root/LogTGN')


from math import e
import os
import torch
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
from torch_geometric.data import TemporalData
from dataloader.utils import calculate_event_weights_from_df, calculate_event_weights_from_df_with_interval
import math
import multiprocessing as mp




# 加载日志结构化数据
def load_log_data(csv_path):
    log_df = pd.read_csv(csv_path)
    if 'Level' in log_df.columns:
        if 'label' in log_df.columns:
            log_df = log_df[['LineId', 'Label', 'Timestamp', 'Level', 'EventId', 'EventTemplate']]
        else:
            log_df = log_df[['LineId', 'Timestamp', 'EventId', 'EventTemplate']]
    else:
        log_df = log_df[['LineId', 'Label', 'Timestamp', 'EventId', 'EventTemplate']]

    return log_df


def load_HDFS_remove_duplicate(csv_path):
    return pd.read_csv(csv_path)


def load_HDFS_log_data(csv_path):
    # 读取的时候将 Date 和 Time 列作为str读入
    log_df = pd.read_csv(csv_path, dtype={'Date': str, 'Time': str})
    
    # 将 Date 和 Time 列合并为一个字符串
    log_df['Datetime'] = '20' + log_df['Date'] + log_df['Time']  # 假设年份是 2008 年

    # 使用 pd.to_datetime 函数将合并后的字符串转换为 datetime 对象
    log_df['Datetime'] = pd.to_datetime(log_df['Datetime'], format='%Y%m%d%H%M%S')

    # 将 datetime 对象转换为 Unix 时间戳
    log_df['Timestamp'] = log_df['Datetime'].astype(int) // 10**9
    
    # Content列是一个字符串，以空格间隔，提取其中以'blk'开头的第一个字符串作为BlockId列
    log_df['BlockId'] = log_df['Content'].str.extract(r'(blk_-?\d+)')
    
    abl = '/root/LogTGN/data/HDFS/preprocessed/anomaly_label.csv'
    al = pd.read_csv(abl)
    # 根据BlockId列的值在abl中查找对应的Label列，如果是BlockId列对应的Label列是Normal则计为'-',如果是Anomaly则计为'Anomaly'，写入log_df的Label列  
    log_df['Label'] = log_df['BlockId'].map(al.set_index('BlockId')['Label'])
    
    log_df = log_df[['LineId', 'Label', 'Datetime', 'Timestamp', 'EventId', 'EventTemplate']]
    return log_df


def get_event_level_from_log_template(template_log_df, event_id):
    return template_log_df[template_log_df['EventId'] == event_id]['Level'].iloc[0]
    

# 连续模板日志去重
def remove_duplicate(log_df):
    data = []
    dic = {}
    for it in log_df.columns:
        dic[it] = log_df[it].iloc[0]
    data.append(dic)
    print(f"data = {data}")
    
    for i in range(len(log_df) - 1):
        # if i > 100000:
        #     break
        dic1, dic2 = {}, {}
        for it in log_df.columns:
            dic1[it] = log_df[it].iloc[i]
            dic2[it] = log_df[it].iloc[i + 1]
        if dic1['EventId'] == dic2['EventId']:
            # print(f"duplicate dict = {dic1}")
            continue
        else:
            # print(f"append dict = {dic2}")
            data.append(dic2)
        
    
    data = pd.DataFrame(data)
    # with open('/root/LogTGN/dataloader/dfeventid.txt', 'a') as f:
    #     f.write("EventId sequence:\n")
    #     f.write(", ".join(data['EventId'].astype(str).tolist()) + "\n")
        # f.write( data['EventId'].tolist())
    # print("EventId sequence:", data['EventId'].tolist())
    print(f"remove duplicate done, data size = {len(data)}")
    return data


# 6. 加载事件向量
def load_event_vectors(vec_path):
    data = torch.load(vec_path)
    # print(data)
    event_vec_df = pd.DataFrame({
        'EventId': data['EventId'],
        'EventTemplate': data['EventTemplate'],
        'EventVector': [vec.tolist() for vec in data['EventVector']]  # 将 tensor 转为列表
    })
    event_vec_df = event_vec_df.set_index('EventTemplate')
    return event_vec_df


def get_event_vector(event_template, event_vec_df):
    return torch.tensor(event_vec_df.loc[event_template, 'EventVector'], dtype=torch.float)


# 7. 计算余弦相似度
def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))

# 构建 TemporalData


def process_chunk(chunk, event_vec_df, template_df, interval, event_id_to_node_id, node_id_counter, edge_weights):
    src_nodes = [[] for _ in range(interval)]
    dst_nodes = [[] for _ in range(interval)]
    timestamps = [[] for _ in range(interval)]
    msg_features = [[] for _ in range(interval)]
    labels = [[] for _ in range(interval)]
    levels = [[] for _ in range(interval)]

    for i in range(interval, len(chunk)):
        event, event_id, vec = [chunk.iloc[i]], [], []
        for j in range(interval + 1):
            event.append(chunk.iloc[i - j])
            event_id.append(chunk.iloc[i - j]['EventId'])
            vec.append(get_event_vector(chunk.iloc[i - j]['EventTemplate'], event_vec_df))
            if event_id[j] not in event_id_to_node_id:
                event_id_to_node_id[event_id[j]] = node_id_counter
                node_id_counter += 1

        if 'Level' in template_df.columns:
            if template_df[template_df['EventId'] == event_id[0]]['Level'].empty:
                continue

        for j in range(interval):
            frequency_score = edge_weights[j][event_id[j+1] + '@' + event_id[0]]
            similarity = cosine_similarity(vec[j+1], vec[0]).item()
            dst_node_id = event_id_to_node_id[event_id[0]]
            src_node_id = event_id_to_node_id[event_id[j+1]]
            src_nodes[j].append(src_node_id)
            dst_nodes[j].append(dst_node_id)
            timestamps[j].append((event[0]['Timestamp']))
            # try:
            #     interval_feature = math.log((int(event[0]['Timestamp']) - int(event[j+1]['Timestamp'])) / 1e9 + 1e-6)
            # except ValueError:
            #     continue
            try:
                interval_feature = math.log((event[0]['Timestamp'] - event[j+1]['Timestamp']) / 1e9 + 1e-6)
            except ValueError:
                print(f"event[0]['Timestamp'] = {event[0]['Timestamp']}, event[j+1]['Timestamp'] = {event[j+1]['Timestamp']}")
                print()
            # interval_feature = math.log((event[0]['Timestamp'] - event[j+1]['Timestamp']) / 1e9 + 1e-6)   # 两个事件之间的时间间隔
            msg_features[j].append([similarity, frequency_score, interval_feature])
            lb = 0 if event[0]['Label'] == 'Normal' else 1
            labels[j].append(lb)
            lv = template_df[template_df['EventId'] == event_id[0]]['Level'].iloc[0] if 'Level' in template_df.columns else 'UNKNOWN'
            levels[j].append(0 if lv == 'INFO' else -1 if lv == 'UNKNOWN' else 1)

    return src_nodes, dst_nodes, timestamps, msg_features, labels, levels



def as_build_temporal_data(dataset, data, event_vec_df, template_df, interval=1):
    src_nodes = [[] for _ in range(interval)]
    dst_nodes = [[] for _ in range(interval)]
    timestamps = [[] for _ in range(interval)]
    msg_features = [[] for _ in range(interval)]
    labels = [[] for _ in range(interval)]
    levels = [[] for _ in range(interval)]

    # 使用一个字典来映射EventId -> NodeId
    event_id_to_node_id = {}
    node_id_counter = 0
    
    # 计算统计频率得分
    _, edge_weights = calculate_event_weights_from_df_with_interval(data, 'EventId', interval=interval)
    # print(edge_weights)
    
    num_chunks = 20
    chunks = np.array_split(data, num_chunks)
    pool = mp.Pool(num_chunks)
    results = [pool.apply_async(process_chunk, args=(chunk, event_vec_df, template_df, interval, event_id_to_node_id, node_id_counter, edge_weights)) for chunk in chunks]
    pool.close()
    pool.join()

    src_nodes, dst_nodes, timestamps, msg_features, labels, levels = [[] for _ in range(interval)], [[] for _ in range(interval)], [[] for _ in range(interval)], [[] for _ in range(interval)], [[] for _ in range(interval)], [[] for _ in range(interval)]
    for result in results:
        res_src_nodes, res_dst_nodes, res_timestamps, res_msg_features, res_labels, res_levels = result.get()
        for j in range(interval):
            src_nodes[j].extend(res_src_nodes[j])
            dst_nodes[j].extend(res_dst_nodes[j])
            timestamps[j].extend(res_timestamps[j])
            msg_features[j].extend(res_msg_features[j])
            labels[j].extend(res_labels[j])
            levels[j].extend(res_levels[j])
        print(f'intervel 0: src_nodes = {len(src_nodes[0])}, dst_nodes = {len(dst_nodes[0])}, timestamps = {len(timestamps[0])}, msg_features = {len(msg_features[0])}, labels = {len(labels[0])}, levels = {len(levels[0])}')
        print(f'intervel 1: src_nodes = {len(src_nodes[1])}, dst_nodes = {len(dst_nodes[1])}, timestamps = {len(timestamps[1])}, msg_features = {len(msg_features[1])}, labels = {len(labels[1])}, levels = {len(levels[1])}')
            
    print(len(src_nodes))
    temporal_datas = []
    for j in range(interval):
        temporal_data = TemporalData(
            src=torch.tensor(src_nodes[j], dtype=torch.long),
            dst=torch.tensor(dst_nodes[j], dtype=torch.long),
            t=torch.tensor(timestamps[j], dtype=torch.long),
            msg=torch.tensor(msg_features[j], dtype=torch.float),
            y=torch.tensor(labels[j], dtype=torch.long),
            level = torch.tensor(levels[j], dtype=torch.long)   # TODO日志级别 
        )
        temporal_datas.append(temporal_data)
    return temporal_datas


def build_temporal_data_with_interval(dataset, data, event_vec_df, template_df, interval=1):
    """构建间隔从1~interval的TemporalData

    Args:
        data (_type_): _description_
        event_vec_df (_type_): _description_
        interval (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    src_nodes = [[] for _ in range(interval)]
    dst_nodes = [[] for _ in range(interval)]
    timestamps = [[] for _ in range(interval)]
    msg_features = [[] for _ in range(interval)]
    labels = [[] for _ in range(interval)]
    levels = [[] for _ in range(interval)]

    # 使用一个字典来映射EventId -> NodeId
    event_id_to_node_id = {}
    node_id_counter = 0
    
    # 计算统计频率得分
    _, edge_weights = calculate_event_weights_from_df_with_interval(data, 'EventId', interval=interval)
    # print(edge_weights)

    
    for i in range(interval, len(data)):
        print(f"pos={i}")
        
        # if i==14703:
        #     import pdb
        #     pdb.set_trace()
            
        event, event_id, vec = [data.iloc[i]], [], []
        # 新增第i个事件和前interval个事件
        # 若interval=2，i=2，增加的是第2，1，0个事件
        for j in range(interval + 1):
            
            event.append(data.iloc[i - j])
            event_id.append(data.iloc[i - j]['EventId'])
            vec.append(get_event_vector(data.iloc[i - j]['EventTemplate'], event_vec_df))
            # 如果EventId没有出现过，创建新的节点ID，构建到节点ID的映射
            if event_id[j] not in event_id_to_node_id:
                event_id_to_node_id[event_id[j]] = node_id_counter
                node_id_counter += 1
            
        if 'Level' in template_df.columns and template_df[template_df['EventId'] == event_id[0]]['Level'].empty:
            continue
        
        # 步长为j时的事件对
        # 例如步长为2时，构建第2个事件和第0个事件
        # 实际上由于前面添加event的时候时先添加的iloc在后面的行，所以这里src应该时较大的index
        for j in range(interval):
                        
            frequency_score = edge_weights[j][event_id[j+1] + '@' + event_id[0]]

            # 计算余弦相似度
            similarity = cosine_similarity(vec[j+1], vec[0]).item()
            # 获取节点ID
            dst_node_id = event_id_to_node_id[event_id[0]]
            src_node_id = event_id_to_node_id[event_id[j+1]]
            # print(f"{j}")
            # print((src_nodes))
            src_nodes[j].append(src_node_id)
            dst_nodes[j].append(dst_node_id)
            timestamps[j].append(int(event[0]['Timestamp']))  # 使用 Timestamp 列
            # print(timestamps[j])
            # event[0]['Timestamp']转为int
            # try:
            #     interval_feature = math.log((int(event[0]['Timestamp']) - int(event[j+1]['Timestamp'])) / 1e9 + 1e-6)   # 两个事件之间的时间间隔
            # except ValueError:
            #     continue
            interval_feature = math.log((event[0]['Timestamp'] - event[j+1]['Timestamp']) / 1e9 + 1e-6)   # 两个事件之间的时间间隔
            msg_features[j].append([similarity, frequency_score, interval_feature])   # 边feature
            
            if dataset == 'HDFS':
                lb = 0 if event[0]['Label'] == 'Normal' else 1
            else:
                lb = 0 if event[0]['Label'] == '-' else 1
            labels[j].append(lb)
            
            lv = ''
            if 'level' in event[0]: # level in BGL
                lv = event[0]['level']
            elif 'Level' in template_df.columns:                   # level in other datasets
                lv = template_df[template_df['EventId'] == event_id[0]]['Level'].iloc[0] 
            else:
                lv = 'UNKNOWN'
                
            if lv == 'INFO': 
                levels[j].append(0)
            elif lv == 'UNKNOWN':
                levels[j].append(-1)
            else:
                levels[j].append(1)    
                
            with open('/root/LogTGN/dataloader/log.txt', 'a') as f:
                f.write(f"src = {event_id_to_node_id[event_id[j+1]]},\t dst = {event_id_to_node_id[event_id[0]]}, \ttime = {event[interval-1]['Timestamp']}, \tmsg = {similarity, frequency_score}, \tlabel = {lb}\n")
          
    print(len(src_nodes))
    temporal_datas = []
    for j in range(interval):
        temporal_data = TemporalData(
            src=torch.tensor(src_nodes[j], dtype=torch.long),
            dst=torch.tensor(dst_nodes[j], dtype=torch.long),
            t=torch.tensor(timestamps[j], dtype=torch.long),
            msg=torch.tensor(msg_features[j], dtype=torch.float),
            y=torch.tensor(labels[j], dtype=torch.long),
            level = torch.tensor(levels[j], dtype=torch.long)   # TODO日志级别 
        )
        temporal_datas.append(temporal_data)
    return temporal_datas


# 9. 保存 TemporalData
def save_temporal_data(dataset, data, event_vec_df, template_df, save_path, interval=1, file_name='temporal_data.pt'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data = remove_duplicate(data) 
    data.to_csv(os.path.join(save_path, "remove_duplicate.csv"), index=False)
    
    # temporal_data = build_temporal_data_with_interval(dataset, data, event_vec_df, template_df, interval)
    temporal_data = as_build_temporal_data(dataset, data, event_vec_df, template_df, interval)
    torch.save(temporal_data, os.path.join(save_path, file_name))


# 主函数
def main(dataset, structure_save_path, template_save_path, vec_save_path, save_file_name, save_dir, interval=1):
    if dataset == 'HDFS':
        if 'remove_duplicate' in structure_save_path:
            log_df = load_HDFS_remove_duplicate(structure_save_path)
        else:
            log_df = load_HDFS_log_data(structure_save_path)
    else:
        log_df = load_log_data(structure_save_path)

    print("日志数据已加载！")
    print(log_df)
    # 加载事件向量
    event_vec_df = load_event_vectors(vec_save_path)
    template_df = pd.read_csv(template_save_path)

    # 保存 TemporalData 到指定目录
    save_temporal_data(dataset, log_df, event_vec_df, template_df, interval=interval, save_path=os.path.join(save_dir), file_name=save_file_name)
    

    print("TemporalData 已保存到指定目录！")


# 运行主程序
if __name__ == "__main__":
    interval =           2
    dataset =            "HDFS"
    structure_save_path =f'/root/LogTGN/data/{dataset}/parser/{dataset}.log_structured.csv'
    template_save_path = f'/root/LogTGN/data/{dataset}/parser/{dataset}.log_templates.csv'
    vec_save_path =      f'/root/LogTGN/data/{dataset}/parser/vec.pt'
    save_file_name =     f'temporal_data_remove_duplicate_interval{interval}.pt'
    save_dir =           f'/root/LogTGN/data/{dataset}/TemporalData'
    # structure_save_path =f"/root/LogTGN/data/{dataset}/TemporalData/remove_duplicate.csv"
    main(dataset, structure_save_path, template_save_path, vec_save_path, interval=interval, save_dir=save_dir, save_file_name=save_file_name)
    