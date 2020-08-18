import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from simple_ntc.bert_trainer_novaild import BertTrainer as Trainer  # bert_trainer_novaild_rere
from simple_ntc.data_loader import BertDataset, TokenizerWrapper


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--txt_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--test_fn', required=True)   
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base')
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config


def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []
        for idx, line in enumerate(lines):
            if idx == 0:
#                 print(line) # header 지우기
                continue
            if line.strip() != '':
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]

    return labels, texts


def get_loaders(fn, fn2, tokenizer):
    # Get list of labels and list of texts.
    labels, texts = read_text(fn)
    labels_t, texts_t = read_text(fn2) # test tsv 
    
    len_train = len(labels)
    len_test = len(labels_t)

    # Generate label to index map.
    unique_labels = sorted(set(labels_t)) #list(set(labels_t))
    print('|unique_labels| =', len(unique_labels)) ##
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))
    
    ## test set : Convert label -> integer
    labels_t = list(map(label_to_index.get, labels_t))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
#     idx = int(len(texts) * .8)

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        BertDataset(texts, labels), # 전체 train
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn= TokenizerWrapper(tokenizer, config.max_length).collate, # { 'text' , 'input_ids', 'attention_mask', 'labels' }'
    )
    
    # test_loader 대신하여 valid loader 구조 차용
    valid_loader = DataLoader(
        BertDataset(texts_t, labels_t), ## 수정
        batch_size=config.batch_size,
        collate_fn=TokenizerWrapper(tokenizer, config.max_length).collate,
    ) # {'text' :[], 'input_ids': tensor([[]]) , 'attention_mask': tensor([[]]) , 'labels': tensor([0, 0]) }
        
    return train_loader, valid_loader, index_to_label, len_train, len_test



# 모델 
def main(config):
    # Get pretrained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label, len_train, len_test = get_loaders(config.train_fn, config.test_fn,  tokenizer)
    # {'text' :[], 'input_ids': tensor([[]]) , 'attention_mask': tensor([[]]) , 'labels': tensor([0, 0]) }
    
    print(
        '|train_all| =', len_train ,
        '|test| =', len_test,
    )
    print(config)
    print('========= pretrained_model_name {} ========='.format(config.pretrained_model_name))
    print('========= model_fn {} ========='.format(config.model_fn))
    print('========= batch_size {} ========='.format(config.batch_size))
    print('========= max_length {} ========='.format(config.max_length))
    print('========= learning rate {} ========='.format(config.lr))
    with open(config.txt_fn, "a") as f:
        f.write('========= config ========= \n|config| : {} \n ========= learning rate {} ========= \n |pretrained_model_name| : {} \n |model_fn| : {} \n |batch_size| : {} \n |max_length| : {} \n |warmup_ratio| : {} \n |adam_epsilon| : {} \n '.format(config ,config.lr, config.pretrained_model_name ,config.model_fn , config.batch_size , config.max_length , config.warmup_ratio, config.adam_epsilon  ))

    # Get pretrained model with specified softmax layer.
    model = BertForSequenceClassification.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_label)
    )
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.lr,
        eps=config.adam_epsilon
    )
    
    # By default, model has softmax layer, not log-softmax layer.
    # Therefore, we need CrossEntropyLoss, not NLLLoss.
    crit = nn.CrossEntropyLoss()

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)
    model1, model2, model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        index_to_label,
    )

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert' : model.state_dict(),
        'bert_bm_loss': model1,
        'bert_bm_acc': model2,
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn) # :저장될 파일명 

    
if __name__ == '__main__':
    config = define_argparser()
    main(config)
