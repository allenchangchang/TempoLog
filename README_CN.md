# TempoLog

## Abstract

The raw log messages are formatted to log templates and represent each template as a dense semantic vector. Then, we construct a multi-scale CTDG based on varying hops of template sequence. Subsequently, we design a semantic-aware TempoLog model to represent the template feature at certain timestamps. Finally, a link prediction model determines whether an edge exists between two nodes, enabling end-to-end training.

By leveraging CTDGs, we eliminate the need for fixed-size windows, enabling the model to capture temporal and contextual dependencies between events dynamically. Additionally, the semantic-aware TempoLog model ensures event-level anomaly detection by learning patterns across varying hops of event interactions. This design directly addresses context bias and fuzzy localization, providing a more accurate, interpretable, and efficient solution for anomaly detection in discrete event sequences. 

![alt text](overview.png)

## Demo

本文件将以BGL数据集为例介绍如何运行整个项目。

### 1. LogParser

首先下载原始数据集放置在 ```./data/BGL```目录下。
然后在```./dataloader```目录下运行```parser.py```文件（注意修改对应的数据集），这会在数据集目录下生成一个```parser```目录，存放处理后的数据```./data/BGL/parser/BGL.log_structured.csv```和```./data/BGL/parser/BGL.log_templates.csv```。

### 2. bert2vec

运行```./dataloader/bert2vev.py```，在数据目录下生成```vec.pt```文件.

```
#  数据格式如下
data_dict = {
    'EventId': vec_df['EventId'].values,
    'EventTemplate': vec_df['EventTemplate'].values,
    'EventVector': event_vector_tensor
}
```

### 3. dataloader

运行```temporaldataloader.py```文件（注意修改数据集名称）。
设置interval的值来确定生成数据集的'H'。
初次运行时需要确保```save_temporal_data```函数中```data = remove_duplicate(data) ```能正常运行。
该文件默认采用多线程处理日志数据，同时也提供了单线程接口。

### 4. 模型训练

运行```./test/multi_tgn_run.py```进行训练。