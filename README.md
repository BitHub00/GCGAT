# Graph Convolutional with Graph Attention Network for Matrix Completion(GCGAT)
## 代码结构
	.
	|-- README.md
	|-- ml-100k (原始数据)
	|-- weights (存储模型)
	|-- data (预处理后的数据)
	|-- scripts
	  -- dataset.py 
	  -- train.py 
	  -- model.py
	  -- utils.py
	  -- loss.py

## 项目介绍
深度学习大作业的代码

## 依赖环境
```bash
python==3.7
pandas==0.24.2
matplotlib==2.2.2
argparse==1.1
tqdm==4.31.1
tensorboardX==1.7
numpy==1.17.3
torch==1.3.1
networkx==2.4
folium==0.10.1
```

## 运行
GCMC: python ./scripts/train.py
GCGAT: python ./scripts/train.py --use_GAT 1

