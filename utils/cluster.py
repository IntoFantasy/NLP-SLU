from utils.example import Example


# 获取语义框架
def slot2sem(slots):
    semantic = []
    for slot in slots:
        semantic.append(Example.label_vocab.convert_tag_to_idx(f'B-{slot}'))
    return tuple(semantic)


class Cluster:
    def __init__(self):
        self.dataset = []
        self.size = 0

    def append(self, data):
        self.dataset.append(data)
        self.size += 1


class ClusterGroup:
    def __init__(self, dataset):
        # {Tuple: Cluster}
        self.index = {}
        self.cluster_num = 0
        self.raw_dataset = dataset

    def add_data(self, data):
        semantic = slot2sem(data.slot)
        if semantic not in self.index:
            self.index[semantic] = Cluster()
            self.cluster_num += 1
        self.index[semantic].append(data)

    def semanticCluster(self):
        for data in self.raw_dataset:
            self.add_data(data)

