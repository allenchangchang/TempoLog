'''
使用bert把log_TEMPLATES转化成vec
'''

from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
import numpy as np

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

DEBUG = False

tokenizer = BertTokenizer.from_pretrained('/root/bert-large-cased')
model = BertModel.from_pretrained("/root/bert-large-cased").to(device)

dataset_name = "HDFS"
df = pd.read_csv(f'/root/LogTGN/data/{dataset_name}/parser/{dataset_name}.log_templates.csv')

i=1
# 定义一个函数将文本转化为向量
def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    if DEBUG:
        global i
        i+=1
        print(i)
    with torch.no_grad():
        outputs = model(**inputs)
    ans = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
    # print(len(ans))
    return ans


# 将EventTemplate列转化为向量
df['EventVector'] = df['EventTemplate'].apply(text_to_vector)

# 提取EventId, EventTemplate和EventVector列
vec_df = df[['EventId', 'EventTemplate', 'EventVector']]

# 确保 EventVector 是数值数组
event_vector_array = np.stack(vec_df['EventVector'].values).astype(np.float32)
# 转换为 PyTorch 张量
event_vector_tensor = torch.tensor(event_vector_array)

# 创建一个字典来存储这些数据
data_dict = {
    'EventId': vec_df['EventId'].values,
    'EventTemplate': vec_df['EventTemplate'].values,
    'EventVector': event_vector_tensor
}

# 保存到 .pt 文件
torch.save(data_dict, f'/root/LogTGN/data/{dataset_name}/parser/vec.pt')


'''
# 读取pt文件步骤
data = torch.load('data/vec.pt')

# 访问数据
event_ids = data['EventId']
event_templates = data['EventTemplate']
event_vectors = data['EventVector']


'''