import time
import os
import random
import argparse
import hashlib

from collections import OrderedDict
import pickle
import torch
from tqdm import tqdm,trange
import numpy as np
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch import distributed

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
    parser.add_argument("--dataset_tag",default='OntoNotes4.0',choices=['ChineseMSRA','en_ace2004','en_ace2005'],)
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
    parser.add_argument("--loss_sampler_epoch",default=10000000,type=int,help="在第几个epoch后启动loss sampler")
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
    parser.add_argument("--reload",action="store_true",help="强制更新缓存的数据，重新开始")
    args = parser.parse_args()
    return args


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()#进程数
    return rt

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
        #print("training steps:",num_training_steps,"warmup_steps:",warmup_steps)
    train_batchs = len(train_dataloader)
    #print("device {} start training".format(device))
    # 模型的id
    mid = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    if args.local_rank in [0,-1]:
        log_dir='./log/{}/{}'.format(args.dataset_tag,mid)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    save_interval = train_batchs//8
    for epoch in range(args.epochs):
        start_time = time.time()
        #print("#############","device:",device," epoch:",epoch,"#############")
        if args.local_rank<1:
            print("#############epoch:",epoch,"#############")
        time.sleep(0.2)
        if args.local_rank!=-1:
            train_dataloader.sampler.set_epoch(epoch)
        tqdm_train_dataloader = tqdm(train_dataloader,ncols=200,desc="batch")
        local_loss = [1e-6]*len(tqdm_train_dataloader)
        if args.local_rank!=-1:
            global_loss = [torch.full([len(train_dataloader)],-1,dtype=torch.float).to(device) for i in range(distributed.get_world_size())]
        for i,batch in enumerate(tqdm_train_dataloader):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            text, mask, segment_id, start, end, span_tensor= batch['text'],batch['mask'], batch['segment_id'],\
                                                             batch['start'],batch['end'],batch['span_tensor']
            text, mask, segment_id, start, end, span_tensor = text.to(device), mask.to(device), segment_id.to(device),\
                                                            start.to(device), end.to(device), span_tensor.to(device)
            loss,all_loss= model(text,mask,segment_id,start,end, span_tensor)
            loss.backward()
            local_loss[i]=loss.item()
            lr = optimizer.param_groups[0]['lr']
            parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in parameters])).item()
            lr_grad_norm = {'grad_norm':grad_norm,'lr':lr}
            if args.max_grad_norm>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if args.warmup_ratio>=0:
                scheduler.step()
            if (i+1)%save_interval==0 and args.local_rank in [0,-1] and not args.not_store:
                if not os.path.exists('./checkpoints/{}/{}'.format(args.dataset_tag,mid)):
                    os.makedirs('./checkpoints/{}/{}'.format(args.dataset_tag,mid))
                if (i+1)==save_interval and epoch==0:
                    pickle.dump(args,open('./checkpoints/%s/%s/args'%(args.dataset_tag,mid),'wb'))
                if hasattr(model,'module'):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save({"epoch":epoch,'state_dict':state_dict},"./checkpoints/%s/%s/checkpoint_%d_%d.cpt"%(args.dataset_tag,mid,epoch,(i+1)//save_interval))
                print("model saved")
            if args.local_rank!=-1:
                reduced_loss = reduce_tensor(loss.data)
            if args.local_rank in [0,-1]:
                #tensorboard在分布式的情况可能有问题
                writer.add_scalars("loss_%s"%mid,all_loss,i+epoch*len(train_dataloader))
                writer.add_scalars("lr_grad_%s"%mid,lr_grad_norm,i+epoch*len(train_dataloader))
                writer.flush()
            postfix_str = "norm:{:.2e},lr:{:.2e},loss:{:.2e},span:{:.2e},start:{:.2e},end:{:.2e},cls:{:.2e}".format(lr_grad_norm['grad_norm'],lr_grad_norm['lr'],all_loss['loss'],all_loss['span_loss'],all_loss['start_loss'],all_loss['end_loss'],all_loss['cls_loss'])
            tqdm_train_dataloader.set_postfix_str(postfix_str)
        if args.local_rank in [0,-1] and not args.not_store:
            if not os.path.exists('./checkpoints/{}/{}'.format(args.dataset_tag,mid)):
                os.makedirs('./checkpoints/{}/{}'.format(args.dataset_tag,mid))
            if epoch==0:
                pickle.dump(args,open('./checkpoints/%s/%s/args'%(args.dataset_tag,mid),'wb'))
            if hasattr(model,'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({"epoch":epoch,'state_dict':state_dict},"./checkpoints/%s/%s/checkpoint_%d.cpt"%(args.dataset_tag,mid,epoch))
            print("model saved")
            #distributed.barrier()
        if args.eval and args.local_rank in [0,-1]:
            print("\n",end="")
            p,r,f = evaluation(model,dev_dataloader)
            print("precision:{:.2f} recall:{:.2f} f1:{:.2f}".format(p,r,f))
            writer.add_scalars("score_%s"%mid,{"precision":p,"recall":r,'f1':f},epoch)
            writer.flush()
            model.train()#评估完之后需要关闭model.eval()
        if args.local_rank!=-1:
            distributed.barrier()
            if (epoch+1)>=args.loss_sampler_epoch:#这里是下一个epoch
                distributed.all_gather(global_loss,torch.tensor(local_loss).to(device))
                train_dataloader.sampler.set_loss(global_loss)
    if args.local_rank in [0, -1]:
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
        hash_path = hashlib.md5('{}_{}_{}_{}_{}'.format(args.pretrained_model_name_or_path,args.train_path,args.train_batch, args.local_rank!=-1, args.allow_impossible).encode()).hexdigest()
        hash_path1 = os.path.join(os.path.split(args.train_path)[0],hash_path)
        if not os.path.exists(hash_path1) or  args.reload:
            train_dataloader = load_data(args.pretrained_model_name_or_path,args.train_path,args.train_batch, args.local_rank!=-1, args.allow_impossible)
            pickle.dump(train_dataloader,open(hash_path1,'wb'))
        else:
            train_dataloader = pickle.load(open(hash_path1,"rb"))
    else:
        hash_path = hashlib.md5('{}_{}_{}_{}'.format(args.pretrained_model_name_or_path, args.train_path,args.max_tokens, args.allow_impossible).encode()).hexdigest()
        hash_path1 = os.path.join(os.path.split(args.train_path)[0],hash_path)
        if not os.path.exists(hash_path1) or args.reload:
            train_dataloader = dist_load_data(args.pretrained_model_name_or_path, args.train_path,args.max_tokens, args.allow_impossible)
            if args.local_rank<1:
                pickle.dump(train_dataloader,open(hash_path1,'wb'))
        else:
            train_dataloader = pickle.load(open(hash_path1,"rb"))
            if args.local_rank!=-1:
                if train_dataloader.sampler.rank!=args.local_rank:
                    print("rank error")
                    train_dataloader.sampler.rank=args.local_rank
    if args.eval:
        hash_path = hashlib.md5('{}_{}_{}'.format(args.pretrained_model_name_or_path,args.dev_path,args.dev_batch).encode()).hexdigest()
        hash_path1 = os.path.join(os.path.split(args.dev_path)[0],hash_path)
        if not os.path.exists(hash_path1) or args.reload:
            dev_dataloader = load_data(args.pretrained_model_name_or_path,args.dev_path,args.dev_batch, False, True)
            if args.local_rank<1:
                pickle.dump(dev_dataloader,open(hash_path1,'wb'))
        else:
            dev_dataloader = pickle.load(open(hash_path1,"rb"))
    else:
        dev_dataloader = None
    print("load data ok...")
    print(len(train_dataloader))
    if args.local_rank!=-1:
        train_dataloader.sampler.set_loss_sampler_epoch(args.loss_sampler_epoch)
    train(args, train_dataloader, dev_dataloader)

