# -*- coding: utf-8 -*-
import sys
import time
import gc

from tqdm import tqdm
from model.seq2seq_model_v2 import *
from utils.args import init_args
from utils.example import Example
from utils.initialization import *
from utils.vocab import PAD
from utils.batch import from_example_list

args = init_args(sys.argv[1:])
set_random_seed(args.seed)
args.device = 0
device = set_torch_device(args.device)

# print(torch.cuda.memory_summary())
start_time = time.time()
train_path = "../data/train.json"
dev_path = "../data/development.json"
word2vec_path = "../word2vec-768.txt"
Example.configuration("../data", train_path=train_path, word2vec_path=word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path, noise=True)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.max_epoch = 10
args.data_path = "../data"
args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
Example.label_vocab.tag2idx['CLS'] = Example.label_vocab.num_tags
Example.label_vocab.idx2tag[74] = 'O'
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)
args.in_hide_dim = 512
args.out_hide_dim = 512
args.in_drop = 0.4
args.out_drop = 0.4
args.out_embed_dim = args.num_tags
args.tmp_dim = args.out_hide_dim

model = Seq2Seq(args).to(device)
model.load_state_dict( torch.load("../model.bin")['model'])


def decode(choice, noise=False):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch, noise)
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


metrics, dev_loss = decode('dev', noise=True)
dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
print('Evaluation: \tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (
        time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
