"""
这个文件的作用就是定义模型，然后在其他文件里面导入
"""
import torch
import torch.nn as nn
from transformers import *

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.device = torch.device('cpu') if config.cpu else torch.device('cuda')
        self.pretrained_model = BertModel.from_pretrained(config.pretrained_model_name_or_path)
        hidden_size = self.pretrained_model.config.hidden_size
        self.start_linear = nn.Linear(hidden_size, 2)
        self.end_linear = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.m = nn.Linear(2*hidden_size, 1)

    def forward(self, input, mask,segment_id):
        """
        用于预测
        Args:
            input:(batch, max_len)
            mask: (batch, max_len)
        Return:
            返回一个batch大小的列表spans，spans[i]代表第输入batch里第i个样本的所有实体的(start,end)位置索引
        """
        #forwrd仅用于预测，没有dropout
        _, rep = self.pretrained_model(input,mask,segment_id)
        start_logits = self.start_linear(rep)
        end_logits = self.end_linear(rep)
        #得到一个batch里面的Istart 和Iend
        start_idxs = []
        end_idxs = []
        for i in range(start_logits.shape[0]):
            start_idx = []
            end_idx = []
            for j in range(start_logits.shape[1]):
                if mask[i][j]!=0:
                    if start_logits[i][j][0]<=start_logits[i][j][1]:
                        start_idx.append(j)
                    if end_logits[i][j][0]<=end_logits[i][j][1]:
                        end_idx.append(j)
            start_idxs.append(start_idx)
            end_idxs.append(end_idx)
        spans = []
        for i in range(input.shape[0]):
            start_idx = start_idxs[i]
            end_idx = end_idxs[i]
            sps = []
            for s in start_idx:
                for e in end_idx:
                    if s<=e and self.m(torch.cat([start_logits[i][s],end_logits[i][e]]))>0:
                        sps.append((s,e))
            spans.append(sps)
        return spans

    def loss(self, input, mask, segment_id, start_target, end_target, spans):
        '''
        :param input: (batch, max_len)
        :param mask: (batch, max_len)
        :param start_target: (batch, max_len)
        :param end_target: (batch, max_len)
        :param spans:类似[[(s1,e1),(s2,e2)],[(s3,e3),(s4,e4)]]
        :return:
        '''
        mask = mask.bool()
        segment_id = segment_id.long()
        start_target = start_target.long()
        end_target = end_target.long()
        rep, _ = self.pretrained_model(input,attention_mask=mask,token_type_ids =segment_id)
        rep = self.dropout(rep)
        start_logits = self.start_linear(rep)#(batch,max_len,2)
        end_logits = self.end_linear(rep)
        batch_size = input.shape[0]
        real_start_logits = torch.cat([start_logits[i][mask[i]]for i in range(batch_size)])
        real_end_logits = torch.cat([end_logits[i][mask[i]] for i in range(batch_size)])
        start_target = start_target.masked_select(mask)
        end_target = end_target.masked_select(mask)
        loss_start = nn.functional.cross_entropy(real_start_logits, start_target)
        loss_end = nn.functional.cross_entropy(real_end_logits, end_target)
        loss_span = []
        for i, sps in enumerate(spans):
            for s,e in sps:
                t = self.m(self.dropout(torch.cat((rep[i][s],rep[i][e]))))
                st = torch.ones(t.shape).to(device=self.device)
                l = nn.functional.binary_cross_entropy_with_logits(t,st)
                loss_span.append(l)
        loss_span = sum(loss_span)
        loss = self.config.alpha*loss_start+self.config.beta*loss_end+self.config.gamma*loss_span
        return loss
