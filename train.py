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
from mydataset import *
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
    parser.add_argument("--pretrained_model_name_or_path",default="./pretrained_models/chinese_roberta_wwm_large_ext_pytorch")
    parser.add_argument("--train_path",default="./datasets/OntoNotes4.0/mrc-ner.train")
    parser.add_argument("--dev_path",default="./datasets/OntoNotes4.0/mrc-ner.dev")
    parser.add_argument("--train_batch",default=5,type=int)
    parser.add_argument("--dev_batch",default=50,type=int)
    parser.add_argument("--epochs",default=10,type=int)
    parser.add_argument("--warmup_steps",default=4000,type=int)
    parser.add_argument("--lr",default=2e-5,type=float)
    parser.add_argument("--weight_decay",default=0.1,type=float)
    parser.add_argument("--dropout_prob",default=0.2,type=float)
    parser.add_argument("--alpha",default=1/3,type=float)
    parser.add_argument("--beta",default=1/3,type=float)
    parser.add_argument("--gamma",default=1/3,type=float)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--device_ids",nargs='+',default=[0,1,2,3],type=int,help="使用的GPU设备")
    parser.add_argument("--eval",action="store_true",help="训练完一个epoch之后是否进行评估")
    parser.add_argument("--seed",default=209,type=int,help="统一的随机数种子")
    args = parser.parse_args()
    return args

def load_data(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    train_dataset = MyDataset(args.train_path, tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size=args.train_batch,collate_fn=collate_fn,shuffle=True)
    if args.eval:
        dev_dataset = MyDataset(args.dev_path, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.train_batch, collate_fn=collate_fn, shuffle=True)
        return train_dataloader, dev_dataloader
    return train_dataloader

def load_data2(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    train_dataset = MyDataset2(args.train_path, tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size=args.train_batch,collate_fn=collate_fn2,shuffle=True)
    if args.eval:
        dev_dataset = MyDataset(args.dev_path, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.train_batch, collate_fn=collate_fn, shuffle=True)
        return train_dataloader, dev_dataloader
    return train_dataloader

def train(args,train_dataloader,dev_dataloader=None):
    '''
    训练模型
    Returns:
    '''
    model = MyModel(args)
    if not args.cpu:
        device = torch.device("cuda")
        model.to(device=device)
        model = torch.nn.DataParallel(model,device_ids=args.device_ids)
    else:
        device = torch.device("cpu")
        model.to(device=device)
    model.train()
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
            {"params":[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay':args.weight_decay},
            {"params":[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0}
        ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    num_training_steps = args.epochs*len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps)
    writer = SummaryWriter(log_dir='./log')
    train_batchs = len(train_dataloader)
    print("start training")
    for epoch in range(args.epochs):
        start_time = time.time()
        print("epoch:",epoch)
        for i,batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            text, mask, start, end,span = batch['text'],batch['mask'],batch['start'],batch['end'],batch['span']
            text, mask, start, end = torch.tensor(text).to(device=device), torch.stack(mask).to(device=device), \
                                     torch.stack(start).to(device=device), torch.stack(end).to(device=device)
            if isinstance(model,torch.nn.DataParallel):
                loss = model.module.loss(text,mask,start,end,span)
            else:
                loss = model.loss(text,mask,start,end,span)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i%100==0:
                current = time.time()
                run_time = (current-start_time)/(60*60)
                remain = run_time*(train_batchs-i-1)/(i+1)
                print("epoch:{}/{} batch:{}/{},当前epoch已经运行{:.2f}h，剩余{:.2f}h".format(epoch + 1, args.epochs, i + 1, train_batchs,
                                                                                    run_time, remain))
            writer.add_scalar("loss",loss.item(),i)
            writer.flush()
        torch.save({"epoch":epoch,'model_state_dict':model.state_dict(),"args":args},"./checkpoint")
    writer.close()

def train2(args,train_dataloader,dev_dataloader=None):
    model = MyModel(args)
    if not args.cpu:
        device = torch.device("cuda")
        model.to(device=device)
        model = torch.nn.DataParallel(model,device_ids=args.device_ids)
    else:
        device = torch.device("cpu")
        model.to(device=device)
    model.train()
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
            {"params":[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay':args.weight_decay},
            {"params":[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0}
        ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    num_training_steps = args.epochs*len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps)
    writer = SummaryWriter(log_dir='./log')
    train_batchs = len(train_dataloader)
    print("start training")
    for epoch in range(args.epochs):
        start_time = time.time()
        print("epoch:",epoch)
        for i,batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            text, mask, segment_id, start, end = batch['text'],batch['mask'], batch['segment_id'],batch['start'],batch['end']
            text, mask, segment_id, start, end = text.to(device=device), mask.to(device=device), segment_id.to(device=device),\
                                     start.to(device=device), end.to(device=device)
            if isinstance(model,torch.nn.DataParallel):
                loss = model.module.loss(text,mask,segment_id,start,end)
            else:
                loss = model.loss(text,mask,segment_id,start,end)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i%(len(train_dataloader)//20)==0:
                current = time.time()
                run_time = (current-start_time)/(60*60)
                remain = run_time*(train_batchs-i-1)/(i+1)
                print("epoch:{}/{} batch:{}/{},当前epoch已经运行{:.2f}h，剩余{:.2f}h".format(epoch + 1, args.epochs, i + 1,
                                                                                    train_batchs, run_time, remain))
                writer.add_scalar("loss",loss.item(),i)
                writer.flush()
        torch.save({"epoch":epoch,'model_state_dict':model.state_dict(),"args":args},"./checkpoint")
        if args.eval:
            p,r,f = evaluation(model,dev_dataloader)
            writer.add_scalar('precision', p, epoch)
            writer.add_scalar('recall', r, epoch)
            writer.add_scalar('f1', f,epoch)
            writer.flush()
    writer.close()

def evaluation(model,dev_dataloader):
    """这部分代码用于训练过程中的评估"""
    model.eval()
    gold = []
    predict = []
    with torch.no_grad:
        for i,batch in enumerate(dev_dataloader):
            text, mask, segment_id, gold_spans = batch['text'], batch['mask'], batch['segment_id'],batch['span']
            pre_spans = model(text,mask,segment_id)
            gold.append((i,gold_spans))
            predict.append((i,pre_spans))
    gold2 = set()
    predict2 = set()
    for g in gold:
        i,gold_spans = g
        for j,gs in enumerate(gold_spans):
            item = (i,j,gs[0],gs[1])
            gold2.add(item)
    for p in predict:
        i,pre_spans = p
        for j, ps in enumerate(pre_spans):
            item = (i,j.ps[0],ps[1])
            predict2.add(item)
    TP = set.intersection(gold2,predict2)
    precision = TP/len(predict2)
    recall = TP/len(gold2)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    print("precision:",precision)
    print("recall:",recall)
    print("f1:",f1)
    return precision,recall,f1

if __name__=="__main__":
    args = args_parser()
    set_seed(args.seed)
    #train_dataloader = load_data(args)
    #train(args, train_dataloader)
    #args.cpu = True
    args.eval = True
    args.train_path = args.dev_path
    train_dataloader,dev_dataloader = load_data2(args)
    train2(args, train_dataloader)
