# -*- coding:utf-8 -*-
import torch
from transformers import BertTokenizer
from transformers import BertModel
from utils.noise_process import *
import warnings
from utils.batch import Batch
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
import torch.nn as nn


class JointBert(nn.Module):
    def __init__(self, config, args):
        super(JointBert, self).__init__()
        self.bert = BertModel.from_pretrained(config)
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(config)
        self.drop = nn.Dropout(args.drop)
        self.linear = nn.Linear(768, args.num_tags)
        self.noise_process = NoiseProcess(args.data_path)
        self.device = "cpu" if args.device == -1 else f"cuda:{args.device}"

    def forward(self, batch):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        outputs = self.bert(input_ids, attention_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = self.linear(self.drop(outputs[0]))

        slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.tag_pad_idx)
        penalty = (1 - batch.tag_mask).unsqueeze(-1).repeat(1, 1, self.args.num_tags) * -1e32
        sequence_output += penalty.to(self.device)
        prob = torch.softmax(sequence_output, dim=-1)
        loss = slot_loss_fct(sequence_output.reshape(-1, self.args.num_tags),
                             batch.tag_ids.reshape(-1).to(self.device))
        return prob, loss

    def decode(self, label_vocab, batch, noise):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if noise:
            self.noise_process.correct(predictions)
        return predictions, labels, loss.cpu().item()

    def from_example_list(self, ex_list, device='cpu', train=True):
        ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
        pad_idx = self.args.pad_idx
        batch = Batch(ex_list, device)
        tag_pad_idx = self.args.tag_pad_idx

        batch.utt = [ex.utt for ex in ex_list]
        input_lens = [len(ex.input_idx) for ex in ex_list]
        max_len = max(input_lens)
        input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]
        batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        batch.lengths = input_lens
        attention_mask = [[1] * len(ex.input_idx) + [0] * (max_len - len(ex.input_idx)) for ex in ex_list]
        batch.attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

        if train:
            batch.labels = [ex.slotvalue for ex in ex_list]
            tag_lens = [len(ex.tag_id) for ex in ex_list]
            max_tag_lens = max(tag_lens)
            tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
            tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]
            batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
            batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
        else:
            batch.labels = None
            batch.tag_ids = None
            batch.tag_mask = None

        return batch

    def dataset_pack(self, dataset):
        pad_idx = self.args.pad_idx
        for ex in dataset:
            for i in range(len(ex.input_idx)):
                if ex.input_idx[i] != pad_idx:
                    ex.input_idx[i] = self.tokenizer.convert_tokens_to_ids(
                        ex.word_vocab.id2word[ex.input_idx[i]])
            # ex.input_idx.insert(0, self.tokenizer.cls_token_id)
            # ex.tag_id.insert(0, self.args.num_tags-2)
            # ex.input_idx.append(self.tokenizer.sep_token_id)
            # ex.tag_id.append(self.args.num_tags-1)


if __name__ == "__main__":
    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # tmp = tokenizer("我")
    # print(tmp)
    # print(tokenizer.cls_token_id)
    # print(tokenizer.sep_token_id)
    # print(tokenizer.pad_token_id)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    #
    # model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")
    print(model)
