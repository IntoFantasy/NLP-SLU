from utils.cluster import *
from utils.example import Example

train_path = "../data/train.json"
dev_path = "../data/development.json"
word2vec_path = "../word2vec-768.txt"
Example.configuration("../data", train_path=train_path, word2vec_path=word2vec_path)

# 语义框架的聚类
train_dataset = Example.load_dataset(train_path)
Clusters = ClusterGroup(train_dataset)
Clusters.semanticCluster()

# 此库扩充
# Example.

print(Clusters.cluster_num)
print(Clusters.index.keys())

