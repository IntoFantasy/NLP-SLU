# -*- coding: utf-8 -*-
import sys
import time
import gc
from tqdm import tqdm

from torch.optim import Adam
from transformers import BertConfig

from model.BERT import *
from utils.args import init_args
from utils.example import Example
from utils.initialization import *
from utils.vocab import PAD

args = init_args(sys.argv[1:])
set_random_seed(args.seed)
args.device = 0
device = set_torch_device(args.device)

start_time = time.time()
train_path = "../data/train.json"
dev_path = "../data/development.json"
word2vec_path = "../word2vec-768.txt"
Example.configuration("../data", train_path=train_path, word2vec_path=word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path, noise=True)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

config = "D:\\我的笔记\\NLP\\bert-chinese\\bert-base-chinese-ner"
args.max_epoch = 10
args.data_path = "../data"
args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
Example.label_vocab.tag2idx['CLS'] = Example.label_vocab.num_tags
Example.label_vocab.tag2idx['SEP'] = Example.label_vocab.num_tags
Example.label_vocab.idx2tag[74] = 'O'
Example.label_vocab.idx2tag[75] = 'O'
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)
args.drop = 0.4
args.lr = 0.00005

model = JointBert(config, args).to(device)
model.dataset_pack(train_dataset)
model.dataset_pack(dev_dataset)


def set_optimizer(model_, args_):
    params = [(n, p) for n, p in model_.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer_ = Adam(grouped_params, lr=args_.lr)
    return optimizer_


def decode(choice, noise=False):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = model.from_example_list(cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch, noise)
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
print('Total training steps: %d' % num_training_steps)
optimizer = set_optimizer(model, args)
nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
train_index, step_size = np.arange(nsamples), args.batch_size
print('Start training ......')

for i in range(args.max_epoch):
    start_time = time.time()
    epoch_loss = 0
    np.random.shuffle(train_index)
    model.train()
    count = 0
    for j in tqdm(range(0, nsamples, step_size)):
        cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
        current_batch = model.from_example_list(cur_dataset, device, train=True)
        outputs, loss = model(current_batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        count += 1
    print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (
        i, time.time() - start_time, epoch_loss / count))
    torch.cuda.empty_cache()
    gc.collect()

    start_time = time.time()
    metrics, dev_loss = decode('dev', noise=True)
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (
        i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
    if dev_acc > best_result['dev_acc']:
        best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result[
            'iter'] = dev_loss, dev_acc, dev_fscore, i
        torch.save({
            'epoch': i, 'model': model.state_dict(),
            'optim': optimizer.state_dict(),
        }, open('../model.bin', 'wb'))
        print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (
            i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (
    best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'],
    best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
