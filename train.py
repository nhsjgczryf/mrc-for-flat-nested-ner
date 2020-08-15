'''
这个文件的功能是：
训练模型
保存训练好的模型
保存训练过程中的历史纪录

'''
import time
from model import MyModel
import mydataset
import torch
import numpy as np
import random
from transformers import *
from transformers.optimization import get_linear_schedule_with_warmup
from model import MyModel
from mydataset import MyDataset
from torch.utils.data import DataLoader
import  argparse
from torch.utils.tensorboard import SummaryWriter
import pickle



"""
model = MyModel(config)
model.train()
optimizer = torch.optim.adamax(model.parameters())
for e in range(len(config.epoches)):
    for batch in dataloader.get_batch_data(config.batch_size):
        model.zero_grad()
        input, mask, start_target, end_target, spans = batch
        loss = model.loss(input,mask,start_target,end_target,spans)
        loss.backward()
        optimizer.step()
"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path",default="bert-base-uncased")
    parser.add_argument("--train_path",default="./datasets/OntoNotes4.0/mrc-ner.train")
    parser.add_argument("--dev_path",default="./datasets/OntoNotes4.0/mrc-ner.dev")
    parser.add_argument("--train_batch",default=15,type=int)
    parser.add_argument("--dev_batch",default=50,type=int)
    parser.add_argument("--max_epoch",default=10,type=int)
    parser.add_argument("--warmup_steps",default=4000,type=int)
    parser.add_argument("--lr",default=2e-5,type=float)
    parser.add_argument("--weight_decay",default=0.1,type=float)
    parser.add_argument("--dropout_prob",default=0.2,type=float)
    parser.add_argument("--device_ids",nargs='+',default=[0,1,2],type=int,help="使用的GPU设备")
    parser.add_argument("--eval",default=True,type=bool,help="训练完一个epoch之后是否进行评估")
    parser.add_argument("--seed",default=209,type=int,help="统一的随机数种子")
    args = parser.parse_args()
    return args

def load_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    train_dataset = MyDataset(args.train_path, tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size=args.train_batch,shuffle=True)
    if args.eval:
        dev_dataset = MyDataset(args.dev_path, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.train_batch, shuffle=True)
        return train_dataloader, dev_dataloader
    return train_dataloader

def train(args,train_dataloader,dev_dataloader):
    '''
    训练模型
    Returns:
    '''
    model = MyModel(args)
    model = torch.nn.DataParallel(model,device_ids=args.device_ids)
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
            {"params":[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay':args.weight_decay},
            {"params":[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0}
        ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps)
    model.train()
    writer = SummaryWriter(log_dir='./log')
    for epoch in range(len(args.max_epoch)):
        for i,batch in enumerate(train_dataloader):


def evaluation():
    '''
    每训练完一个epoch评估一下
    Returns:
    '''


if __name__=="__main__":
    args = args_parser()
    set_seed(args.seed)