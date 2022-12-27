# -*- coding:utf-8 -*-
from utils.noise_process import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, in_hide_dim, out_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, in_hide_dim, bidirectional=True)
        self.fc = nn.Linear(in_hide_dim * 2, out_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src_len, batch_size]
        src = src.transpose(0, 1)
        # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src)).transpose(0, 1)
        # embedded = [src_len, batch_size, emb_dim]
        # in_output = [src_len, batch_size, hid_dim * num_directions]
        # in_hidden = [n_layers * num_directions, batch_size, hid_dim]
        in_output, in_hidden = self.rnn(embedded)
        s = torch.tanh(self.fc(torch.cat((in_hidden[-2, :, :], in_hidden[-1, :, :]), dim=1)))
        return in_output, s


class Attention(nn.Module):
    def __init__(self, in_hid_dim, out_hid_dim, temp_dim):
        super().__init__()
        self.temp_dim = temp_dim
        self.attn = nn.Linear((in_hid_dim * 2) + out_hid_dim, self.temp_dim, bias=False)
        self.v = nn.Linear(self.temp_dim, 1, bias=False)

    def forward(self, s, in_output):
        src_len = in_output.shape[0]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        in_output = in_output.transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((s, in_output), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        # self.embedding = nn.Embedding(output_dim, emb_dim)
        self.embedding = F.one_hot
        self.emb_dim = emb_dim
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, out_input, s, in_output):
        # out_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # in_output = [src_len, batch_size, enc_hid_dim * 2]
        out_input = out_input.unsqueeze(1)  # dec_input = [batch_size, 1]
        embedded = self.embedding(out_input, num_classes=self.emb_dim).transpose(0, 1)
        a = self.attention(s, in_output).unsqueeze(1)
        in_output = in_output.transpose(0, 1)
        c = torch.bmm(a, in_output).transpose(0, 1)
        rnn_input = torch.cat((embedded, c), dim=2)
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))
        embedded = embedded.squeeze(0)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(0)
        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))

        return pred, dec_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        attention = Attention(config.in_hide_dim, config.out_hide_dim, config.tmp_dim)
        self.encoder = Encoder(config.vocab_size, config.embed_size,
                               config.in_hide_dim, config.out_hide_dim, config.in_drop)
        self.decoder = Decoder(config.num_tags, config.out_embed_dim,
                               config.in_hide_dim, config.out_hide_dim, config.out_drop, attention)
        self.device = "cpu" if config.device == -1 else f"cuda:{config.device}"
        print(f"using {self.device} to train")
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=config.tag_pad_idx)
        self.noise_process = NoiseProcess(config.data_path)

    def forward(self, batch):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio is probability to use teacher forcing
        src = batch.input_ids.transpose(0, 1)
        trg = batch.tag_ids.transpose(0, 1)
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        in_output, s = self.encoder(src)
        out_input = trg[0, :]
        for t in range(trg_len):
            out_output, s = self.decoder(out_input, s, in_output)
            outputs[t] = out_output
            # teacher_force = random.random() < teacher_forcing_ratio
            # top = out_output.argmax(1)
            # out_input = trg[t] if teacher_force else top
            out_input = trg[t]
        outputs = outputs.transpose(0, 1)
        trg = trg.transpose(0, 1)
        penalty = (1 - batch.tag_mask).unsqueeze(-1).repeat(1, 1, self.config.num_tags) * -1e32
        outputs += penalty.to(self.device)
        prob = torch.softmax(outputs, dim=-1)
        loss = self.loss_fct(outputs.reshape(-1, self.config.num_tags), trg.reshape(-1).to(self.device))
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
