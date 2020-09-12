"""
这个文件的作用就是定义模型，然后在其他文件里面导入
"""
import torch
import torch.nn as nn
from transformers import BertModel
from tqdm import trange
            

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.pretrained_model = BertModel.from_pretrained(config.pretrained_model_name_or_path)
        hidden_size = self.pretrained_model.config.hidden_size
        self.start_linear = nn.Linear(hidden_size, 2)
        self.end_linear = nn.Linear(hidden_size, 2)
        if config.cls:
            self.cls = nn.Linear(hidden_size,1)
        self.dropout = nn.Dropout(config.dropout_prob)
        if config.span_layer:
            self.m = nn.Sequential(nn.Linear(2*hidden_size,hidden_size),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(config.dropout_prob),
                                   nn.Linear(hidden_size,1))
        else:
            self.m = nn.Linear(2*hidden_size, 1)

    def predict(self, input, mask,segment_id,mask_decode=True):
        """
        用于预测
        Args:
            input:(batch, max_len)
            mask: (batch, max_len)
        Return:
            返回一个batch大小的列表spans，spans[i]代表第输入batch里第i个样本的所有实体的(start,end)位置索引
        """
        #仅用于预测，没有dropout
        #print("device:",input.device)
        #print("input shape:",input.shape)
        mask = mask.bool()
        segment_id = segment_id.long()
        rep, cls_rep = self.pretrained_model(input,mask,segment_id)
        start_logits = self.start_linear(rep)
        end_logits = self.end_linear(rep)
        if mask_decode:
            start_predict = start_logits.argmax(-1)
            end_predict = end_logits.argmax(-1)
            context_mask = (segment_id==1)&mask
            seq_len = rep.shape[1]
            rep_start = rep.unsqueeze(2).expand(-1,-1,seq_len,-1)
            rep_end = rep.unsqueeze(1).expand(-1,seq_len,-1,-1)
            span_rep = torch.cat((rep_start,rep_end),dim=-1)
            span_rep = self.m(self.dropout(span_rep))
            span_rep = span_rep.squeeze(-1)
            #mask掉<0的情形
            span_mask0 = span_rep>=0
            #mask掉非start和end的情况
            span_mask1_start = start_predict.unsqueeze(2).expand(-1,-1,seq_len)#span_mask_start[b][i][j]=start_predict[b][i]
            span_mask1_end = end_predict.unsqueeze(1).expand(-1,seq_len,-1)#span_mask_end[b][i][j]=end_predict[b][j]
            #span_mask1[b][i][j]==True当且仅当start_predict[b][i]和 end_predict[b][j]均为1
            span_mask1 = (span_mask1_start&span_mask1_end).bool()
            #mask掉query中的单词
            span_mask2_start = context_mask.unsqueeze(2).expand(-1,-1,seq_len)#span_mask_start[b][i][j]=context_mask[b][i]
            span_mask2_end = context_mask.unsqueeze(1).expand(-1,seq_len,-1)#span_mask_end[b][i][j]=context_mask[b][j]
            #下面的span_mask和作者的基本一样，但是其实还可以优化，就是我们直接传入span_mask这个参数，并且这个span_mask还可以mask掉末尾的[SEP]
            span_mask2 = span_mask2_start&span_mask2_end
            #以及mask掉soan_mask[b][i][j]中i>j(即start>end的情况)
            span_mask3 = torch.triu(torch.ones(span_rep.shape)).bool().cuda()
            #最终的mask
            #print(span_mask0.device,span_mask1.device,span_mask2.device,span_mask3.device)
            span_mask = ((span_mask0&span_mask1)&span_mask2)&span_mask3
            spans = []
            for b in range(input.shape[0]):
                span = torch.nonzero(span_mask[b])
                spans.append(span.tolist())
        else:
            #这种方法对GPU内存要求不高，训练的时候可以降低GPU的压力
            #得到一个batch里面的Istart 和Iend
            start_idxs = []
            end_idxs = []
            #print("start logits:",start_logits[0].T)
            #print("end logits:",end_logits[0].T)
            #print("get start and end logits")
            for i in range(start_logits.shape[0]):
                start_idx = []
                end_idx = []
                for j in range(start_logits.shape[1]-1):
                    if segment_id[i][j]!=0:
                        if start_logits[i][j][0]<=start_logits[i][j][1]:
                            start_idx.append(j)
                        if end_logits[i][j][0]<=end_logits[i][j][1]:
                            end_idx.append(j)
                start_idxs.append(start_idx)
                end_idxs.append(end_idx)
            #print("start idxs:",start_idxs)
            #print("end idxs:",end_idxs)
            #print("start and end idxs length:",len(start_idxs))
            spans = []
            for i in range(input.shape[0]):
                start_idx = start_idxs[i]
                end_idx = end_idxs[i]
                sps = []
                if not self.config.cls or self.cls(cls_rep)[i]>=0:
                    for s in start_idx:
                        for e in end_idx:
                            #if s<=e:
                            #    print(s,e,self.m(torch.cat([rep[i][s],rep[i][e]])).item())
                            if s<=e and self.m(torch.cat([rep[i][s],rep[i][e]]))>0:
                                sps.append((s,e))
                spans.append(sps)
        #print("spans:",spans[0])
        return spans

    def forward(self, input, mask, segment_id, start_target, end_target, span_tensor):
        '''
        :param input: (batch, max_len)
        :param mask: (batch, max_len)
        :param start_target: (batch, max_len)
        :param end_target: (batch, max_len)
        :return:
        '''
        #if torch.cuda.is_available():
        #    print("In model: input size:",input.shape)
        mask = mask.bool()
        segment_id = segment_id.long()
        #在segment_id用0来pad的情况下，这个context_mask和segment_id是一样的
        context_mask = (segment_id==1)&mask
        start_target = start_target.long()
        end_target = end_target.long()
        span_tensor = span_tensor.long()
        batch_size = input.shape[0]
        self.device = torch.device('cpu') if self.config.cpu else torch.cuda.current_device()

        rep, cls_rep = self.pretrained_model(input,attention_mask=mask,token_type_ids =segment_id)
        rep = self.dropout(rep)
        start_logits = self.start_linear(rep)#(batch,max_len,2)
        end_logits = self.end_linear(rep)
        #real_start_logits = torch.cat([start_logits[i][context_mask[i]]for i in range(batch_size)])
        real_start_logits = start_logits[context_mask]
        #real_end_logits = torch.cat([end_logits[i][context_mask[i]] for i in range(batch_size)])
        real_end_logits = end_logits[context_mask]
        real_start_target = start_target.masked_select(context_mask)
        real_end_target = end_target.masked_select(context_mask)
        loss_start = nn.functional.cross_entropy(real_start_logits, real_start_target,reduction=self.config.reduction)
        loss_end = nn.functional.cross_entropy(real_end_logits, real_end_target,reduction=self.config.reduction)
        if self.config.train_span_method!="full":
            #获取真实的spans
            spans = []
            starts = []
            ends = []
            for i in range(batch_size):
                start = []
                end = []
                for j in range(0, span_tensor.shape[1],2):
                    if span_tensor[i][j]!=-1:
                        start.append(span_tensor[i][j].item())
                        end.append(span_tensor[i][j+1].item())
                assert len(start)==len(end)
                starts.append(start)
                ends.append(end)
                spans.append(list(zip(start, end)))
            #print("get spans:",spans)
            if self.config.train_span_method in ['predict','mix']:
                predict_starts = []
                predict_ends = []
                for i in range(batch_size):
                    predict_start = []
                    predict_end = []
                    for j in range(input.shape[1]-1):
                        if segment_id[i][j]!=0:
                            if start_logits[i][j][0]<=start_logits[i][j][1]:
                                predict_start.append(j)
                            if end_logits[i][j][0]<=end_logits[i][j][1]:
                                predict_end.append(j)
                    predict_starts.append(predict_start)
                    predict_ends.append(predict_end)
            if self.config.train_span_method=='gold':
                all_starts = map(set,starts)
                all_ends = map(set,ends)
            elif self.config.train_span_method=='predict':
                all_starts = map(set,predict_starts)
                all_ends = map(set,predict_ends)
            else:
                all_starts = map(lambda x: set(x[0]+x[1]),zip(starts,predict_starts))
                all_ends = map(lambda x: set(x[0]+x[1]),zip(ends,predict_ends))
            all_starts = list(all_starts)
            all_ends = list(all_ends)
            all_span_target = []
            all_span_logit = []
            for i in range(batch_size):
                for s in all_starts[i]:
                    for e in all_ends[i]:
                        if s<=e:
                            if (s,e) in spans[i]:
                                span_target = torch.tensor([1.],device=self.device)
                            else:
                                span_target = torch.tensor([0.],device=self.device)
                            span_logit = self.m(self.dropout(torch.cat((rep[i][s],rep[i][e]))))
                            all_span_target.append(span_target)
                            all_span_logit.append(span_logit)
            if len(all_span_target)==0:
                loss_span = torch.tensor(0)
            else:
                true_spans = torch.cat(all_span_target)
                span_logits = torch.cat(all_span_logit)
                loss_span = nn.functional.binary_cross_entropy_with_logits(span_logits,true_spans,reduction=self.config.reduction)
        else:
            seq_len = rep.shape[1]
            span_mask_start = context_mask.unsqueeze(2).expand(-1,-1,seq_len)#span_mask_start[b][i][j]=context_mask[b][i]
            span_mask_end = context_mask.unsqueeze(1).expand(-1,seq_len,-1)#span_mask_end[b][i][j]=context_mask[b][j]
            #下面的span_mask和作者的基本一样，但是其实还可以优化，就是我们直接传入span_mask这个参数，并且这个span_mask还可以mask掉末尾的[SEP]
            #以及mask掉soan_mask[b][i][j]中i>j(即start>end的情况)
            span_mask = span_mask_start&span_mask_end
            span_mask = torch.triu(span_mask).bool()
            rep_start = rep.unsqueeze(2).expand(-1,-1,seq_len,-1)
            rep_end = rep.unsqueeze(1).expand(-1,seq_len,-1,-1)
            span_rep = torch.cat((rep_start,rep_end),dim=-1)
            span_rep = span_rep[span_mask]#先mask再计算能降低内存的使用
            span_rep = self.m(self.dropout(span_rep))
            span_rep = span_rep.squeeze(-1)
            span_start_target = start_target.unsqueeze(2).expand(-1,-1,seq_len)#span_mask_start[b][i][j]=start_target[b][i]
            span_end_target = end_target.unsqueeze(1).expand(-1,seq_len,-1)#span_mask_start[b][i][j]=end_target[b][j]
            full_span_target = (span_start_target&span_end_target).float()
            full_span_target = full_span_target[span_mask]
            loss_span = nn.functional.binary_cross_entropy_with_logits(span_rep,full_span_target,reduction=self.config.reduction)
            
        #print("start positive ratio:",sum(real_start_target).item()/len(real_end_target))
        #print("end positive ratio:",sum(real_end_target).item()/len(real_end_target))
        #print("span positive ratio:",sum(all_span_target).item()/len(all_span_target))
        if self.config.cls:
            cls_rep = self.cls(self.dropout(cls_rep))
            cls_target = torch.zeros(cls_rep.shape).to(self.device)
            for i,st in enumerate(start_target):
                if st.sum()>=1:
                    cls_target[i][0]=1
                else:
                    cls_target[i][0]=0
            loss_cls = nn.functional.binary_cross_entropy_with_logits(cls_rep,cls_target,reduction=self.config.reduction)
        if self.config.train_span_method!='full' and len(all_span_target)==0:
            loss_span=torch.tensor(0)
        if not self.config.cls:
            loss_cls=torch.tensor(0)
        loss = self.config.alpha*loss_start+self.config.beta*loss_end+self.config.gamma*loss_span+self.config.theta*loss_cls
        #print("start loss",loss_start.item()," end loss:",loss_end.item()," span loss:",loss_span.item(),"cls loss:",loss_cls.item())
        other_loss = {}
        return loss.view(1),{'loss':loss.item(),"start_loss":loss_start.item(),"end_loss":loss_end.item(),"span_loss":loss_span.item(),"cls_loss":loss_cls.item()}
    
    
class MyModel2(MyModel):
    def predict(self, input, mask,segment_id,mask_decode=True,threshold=-0.1):
        #仅用于预测，没有dropout
        #print('threshold:',threshold)
        #print("device:",input.device)
        #print("input shape:",input.shape)
        mask = mask.bool()
        segment_id = segment_id.long()
        rep, cls_rep = self.pretrained_model(input,mask,segment_id)
        start_logits = self.start_linear(rep)
        end_logits = self.end_linear(rep)
        #这种方法对GPU内存要求不高，训练的时候可以降低GPU的压力
        #得到一个batch里面的Istart 和Iend
        start_idxs = []
        end_idxs = []
        #print("start logits:",start_logits[0].T)
        #print("end logits:",end_logits[0].T)
        #print("get start and end logits")
        for i in range(start_logits.shape[0]):
            start_idx = []
            end_idx = []
            for j in range(start_logits.shape[1]-1):
                if segment_id[i][j]!=0:
                    if start_logits[i][j][0]<=start_logits[i][j][1]:
                        start_idx.append(j)
                    if end_logits[i][j][0]<=end_logits[i][j][1]:
                        end_idx.append(j)
            start_idxs.append(start_idx)
            end_idxs.append(end_idx)
        #print("start idxs:",start_idxs)
        #print("end idxs:",end_idxs)
        #print("start and end idxs length:",len(start_idxs))
        spans = []
        for i in range(input.shape[0]):
            start_idx = start_idxs[i]
            end_idx = end_idxs[i]
            sps = []
            for s in start_idx:
                for e in end_idx:
                    #if s<=e:
                    #    print(s,e,self.m(torch.cat([rep[i][s],rep[i][e]])).item())
                    if s<=e and self.m(torch.cat([rep[i][s],rep[i][e]]))>threshold:
                        sps.append((s,e))
            spans.append(sps)
        #print("spans:",spans[0])
        return spans
