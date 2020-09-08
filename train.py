import time
import os
import random
import argparse

from collections import OrderedDict
import pickle
import torch
import numpy as np
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from mydataset import load_data,dist_load_data
from model import MyModel
from evaluate import get_score,evaluation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",default="./datasets/OntoNotes4.0/mrc-ner.train")
    parser.add_argument("--dev_path",default="./datasets/OntoNotes4.0/mrc-ner.dev")
    parser.add_argument("--pretrained_model_name_or_path",default="./pretrained_models/chinese_roberta_wwm_large_ext_pytorch")
    parser.add_argument("--train_batch",default=4,type=int)
    parser.add_argument("--max_tokens",default=512,type=int,help="这个值应该大于样本长度的最大值")
    parser.add_argument("--dev_batch",default=10,type=int)
    parser.add_argument("--epochs",default=10,type=int)
    parser.add_argument("--train_span_method",default='mix',choices=['gold','predict','mix','full'],
                        help="gold代表Istart和Iend是真实值的集合,predict代表是这两个集合是预测值的集合,mix是代表这两个集合的并集,full代表所有的位置(full会导致span loss存在大大大量负样本，导致模型预测很差)")
    parser.add_argument("--warmup_ratio",default=0.1,type=float)
    parser.add_argument("--lr",default=2e-5,type=float)
    parser.add_argument("--weight_decay",default=0.01,type=float)
    parser.add_argument("--dropout_prob",default=0.1,type=float)
    parser.add_argument("--alpha",default=1,type=float)
    parser.add_argument("--beta",default=1,type=float)
    parser.add_argument("--gamma",default=1,type=float)
    parser.add_argument("--theta",default=1,type=float)
    parser.add_argument("--cls",action="store_true",help="是否启用[CLS]分类判断是否存在当前类型的实体")
    parser.add_argument("--reduction",default='mean',choices=['sum','mean'])
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--log_steps",default=20,type=int)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--max_grad_norm", default=-1, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_impossible",action="store_true",help="是否允许impossible的样本")
    parser.add_argument("--eval",action="store_true",help="训练完一个epoch之后是否进行评估")
    parser.add_argument("--seed",default=209,type=int,help="统一的随机数种子")
    parser.add_argument("--not_store",action="store_true")
    parser.add_argument("--span_layer",action="store_true")
    args = parser.parse_args()
    return args


def train(args,train_dataloader,dev_dataloader=None):
    print(args)
    model = MyModel(args)
    model.train()
    if torch.cuda.is_available():
        local_rank =  args.local_rank
        if local_rank!=-1:
            device = local_rank
            torch.cuda.set_device(local_rank)
            #print('local_rank:',args.local_rank)
            model.to(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],\
                                                              output_device=local_rank,find_unused_parameters=True)
        else:
            device = torch.device("cuda")
            model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device=device)
    #print(device)
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
            {"params":[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay':args.weight_decay},
            {"params":[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0}
        ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    if args.warmup_ratio>=0:
        num_training_steps = args.epochs*len(train_dataloader)
        warmup_steps = args.warmup_ratio*num_training_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
        print("training steps:",num_training_steps,"warmup_steps:",warmup_steps)
    writer = SummaryWriter(log_dir='./log')
    train_batchs = len(train_dataloader)
    print("device {} start training".format(device))
    # 模型的id
    mid = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    if not os.path.exists('./checkpoints/'+mid):
        os.makedirs('./checkpoints/'+mid)
    pickle.dump(args,open('./checkpoints/%s/args'%mid,'wb'))
    save_interval = train_batchs//8
    for epoch in range(args.epochs):
        start_time = time.time()
        print("#############","device:",device," epoch:",epoch,"#############")
        if args.local_rank!=-1:
            train_dataloader.sampler.set_epoch(epoch)
        for i,batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            text, mask, segment_id, start, end, span_tensor= batch['text'],batch['mask'], batch['segment_id'],\
                                                             batch['start'],batch['end'],batch['span_tensor']
            text, mask, segment_id, start, end, span_tensor = text.to(device), mask.to(device), segment_id.to(device),\
                                                            start.to(device), end.to(device), span_tensor.to(device)
            #print("gold span:",batch['span'])
            loss = model(text,mask,segment_id,start,end, span_tensor)
            #loss = loss.mean()
            #print("loss device:",loss.device)
            loss.backward()
            if args.max_grad_norm>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if args.warmup_ratio>=0:
                scheduler.step()
            #if (i+1)%save_interval==0 and args.local_rank in [0,-1] and not args.not_store:
            #    if hasattr(model,'module'):
            #        state_dict = model.module.state_dict()
            #    else:
            #        state_dict = model.state_dict()
            #    torch.save({"epoch":epoch,'state_dict':state_dict},"./checkpoints/%s/checkpoint_%d_%d.cpt"%(mid,epoch,(i+1)//save_interval))
            #    print("model saved")
            if i%args.log_steps==0:
                current = time.time()
                run_time = (current-start_time)/(60*60)
                remain = run_time*(train_batchs-i-1)/(i+1)
                print("device:{} epoch:{}/{} batch:{}/{},loss:{} 当前epoch已经运行{:.2f}h，剩余{:.2f}h".format(device,
                                        epoch + 1, args.epochs, i + 1,train_batchs, loss.item(),run_time, remain))
                writer.add_scalar("loss_%s"%mid,loss.item(),i)
                writer.flush()
        if args.local_rank in [0,-1] and not args.not_store:
            if hasattr(model,'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({"epoch":epoch,'state_dict':state_dict},"./checkpoints/%s/checkpoint_%d.cpt"%(mid,epoch))
            print("model saved")
        if args.eval and args.local_rank in [0,-1]:
            print("############evalutation###############")
            #评估只在
            p,r,f = evaluation(model,dev_dataloader)
            print("precision:{:.2f} recall:{:.2f} f1:{:.2f}".format(p,r,f))
            writer.add_scalar('precision_'+mid, p, epoch)
            writer.add_scalar('recall_'+mid, r, epoch)
            writer.add_scalar('f1_'+mid, f,epoch)
            writer.flush()
            model.train()#评估完之后需要关闭model.eval()
    writer.close()

if __name__=="__main__":
    args = args_parser()
    if not torch.cuda.is_available():
        args.cpu = True
    else:
        args.cpu = False
    set_seed(args.seed)
    if args.local_rank!=-1:
        torch.distributed.init_process_group(backend='nccl')
    if args.debug:
        args.train_path = '/home/wangnan/mrc4ner/datasets/OntoNotes4.0/testdata.json'
        args.dev_path = '/home/wangnan/mrc4ner/datasets/OntoNotes4.0/testdata.json'
    #导入数据
    print("start loading data...")
    if args.local_rank==-1:
        train_dataloader = load_data(args.pretrained_model_name_or_path,args.train_path,args.train_batch, args.local_rank!=-1)
    else:
        train_dataloader = dist_load_data(args.pretrained_model_name_or_path, args.train_path,args.max_tokens, args.allow_impossible)
    if args.eval:
        #dev_dataloader = load_data(args.pretrained_model_name_or_path,args.dev_path,args.dev_batch, args.local_rank!=-1, args.allow_impossible)
        dev_dataloader = load_data(args.pretrained_model_name_or_path,args.dev_path,args.dev_batch, False, True)
    else:
        dev_dataloader = None
    print("load data ok...")
    train(args, train_dataloader, dev_dataloader)

