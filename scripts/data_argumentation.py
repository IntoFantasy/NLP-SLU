# -*- coding:utf-8 -*-
import json

data = json.load(open("../data/train.json", 'r', encoding='UTF-8'))
data_argumentation = []
# tag = ["poi名称", "poi修饰", "poi目标", "起点名称", "起点修饰", "起点目标", "终点名称", "终点修饰", "终点目标", "途经点名称"]
tag = ["终点名称"]
with open("../data/lexicon/poi_name.txt", 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    for poi in lines:
        poi = poi.replace('\n', '')
        for slot in tag:
            data_added = [{
                'utt_id': 1,
                'manual_transcript': poi,
                'asr_1best': poi,
                'semantic': [['inform', slot, poi]]
            }]
            data.append(data_added)
with open("../data/data_argumentation.json", 'w', encoding='UTF-8') as fp:
    json.dump(data, fp, ensure_ascii=False, indent=2)

