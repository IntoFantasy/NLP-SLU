from seq2seq_model_v2 import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, max_utt_len, max_pheno_len, device):
        super().__init__()
        self.device = device
        self.max_utt_len = max_utt_len
        self.max_pheno_len = max_pheno_len
        self.utt_conv = nn.Conv1d(in_channels=self.max_utt_len, out_channels=self.max_utt_len,
                                  kernel_size=self.max_pheno_len)
        self.pheno_conv = nn.Conv1d(in_channels=self.max_pheno_len, out_channels=self.max_pheno_len,
                                    kernel_size=self.max_utt_len)

    def forward(self, utt_output, pheno_output):
        # utt_output = [utt_len, batch, hid_dim * 2]
        # pheno_output = [pheno_len, batch, hid_dim * 2]
        utt_output = utt_output.transpose(0, 1)
        pheno_output = pheno_output.transpose(0, 1)
        batch_size = utt_output.shape[0]
        utt_len = utt_output.shape[1]
        pheno_len = pheno_output.shape[1]
        utt_hid_dim = utt_output.shape[2]
        pheno_hid_dim = pheno_output.shape[2]
        # utt_output = [batch, utt_len, hid_dim * 2]
        # pheno_output = [batch, pheno_len, hid_dim * 2]
        utt_cat = torch.zeros((batch_size, self.max_utt_len-utt_len, utt_hid_dim)).to(self.device)
        pheno_cat = torch.zeros((batch_size, self.max_pheno_len-pheno_len, pheno_hid_dim)).to(self.device)
        utt_output = torch.cat((utt_output, utt_cat), dim=1)
        pheno_output = torch.cat((pheno_output, pheno_cat), dim=1)
        c = torch.zeros((batch_size, self.max_utt_len, self.max_pheno_len)).to(self.device)
        for i in range(self.max_utt_len):
            for j in range(self.max_pheno_len):
                c[:, i, j] = torch.cosine_similarity(utt_output[:, i, :], pheno_output[:, j, :], dim=-1)
        row_attention = F.softmax(self.utt_conv(c).reshape((batch_size, self.max_utt_len))[:, :utt_len], dim=-1)
        col_attention = F.softmax(self.pheno_conv(c.transpose(1, 2)).reshape((batch_size, self.max_pheno_len))
                                  [:, :pheno_len], dim=-1)
        return row_attention, col_attention


if __name__ == "__main__":
    a = torch.rand((10, 32, 20))
    b = torch.rand((12, 32, 20))
    model = CrossAttention(40, 80, 'cpu')
    att1, att2 = model(a, b)
    print(att1.shape)
    print(att2.shape)