# -*- coding: utf-8 -*-
import json
import numpy as np


class NoiseProcess:
    def __init__(self, filePath):
        self.ontology = self.dataLoader(filePath)

    @staticmethod
    def Lev_distance(word1, word2):
        dp = np.array(np.arange(len(word2) + 1))

        for i in range(1, len(word1) + 1):
            temp1 = dp[0]
            dp[0] += 1
            for j in range(1, len(word2) + 1):
                temp2 = dp[j]
                if word1[i - 1] == word2[j - 1]:
                    dp[j] = temp1
                else:
                    dp[j] = min(temp1, min(dp[j - 1], dp[j])) + 1
                temp1 = temp2

        return dp[len(word2)]

    @staticmethod
    def dataLoader(path):
        ont_path = path + '/ontology.json'
        data = json.load(open(ont_path, 'r', encoding='utf-8'))["slots"]
        for key in data.keys():
            if isinstance(data[key], str):
                fp = path + data[key][1:]
                with open(fp, 'r', encoding='utf-8') as f:
                    obj = []
                    lines = f.readlines()
                    for var in lines:
                        var = var.replace('\n', '')
                        obj.append(var)
                    data[key] = obj
        return data

    def correct(self, examples):
        for j in range(len(examples)):
            example = examples[j]
            for i in range(len(example)):
                lis = example[i].split('-')
                slot = lis[1]
                value = lis[2]
                lis[2] = self.find_similarity_value(value, slot)
                example[i] = '-'.join(lis)

    def find_similarity_value(self, value, slot):
        if value in self.ontology[slot]:
            return value
        else:
            sim_value = None
            dis = 1e6
            for slot_value in self.ontology[slot]:
                dis_tmp = self.Lev_distance(value, slot_value)
                sim_value, dis = (slot_value, dis_tmp) if dis_tmp < dis else (sim_value, dis)
            return sim_value


if __name__ == "__main__":
    A = [["inform-终点名称-原火车站"]]
    fp = "../data"
    data = NoiseProcess(fp)
    data.correct(A)
    print(A)

