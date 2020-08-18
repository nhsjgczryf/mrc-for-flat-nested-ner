"""
这个文件的作用是
把数据转换为batch的feature
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import *

def get_batch_data(batch_size):
    """
    :param batch_size:
    :return:
    """
    tokenizer = None
    labels = {'LOC': 0, 'PER': 1, 'GPE': 2, 'ORG': 3}
    # 导入文件
    path = r'C:\Users\DELL\Desktop\mrc4nner\datasets\OntoNotes4.0\mrc-ner.train'
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    # 提取数据
    examples = []
    for d in data:
        if not d['impossible']:
            context = d['context']
            context = context.split()
            query = d['query']
            query = list(query)
            text = ['CLS'] + query + ['SEP'] + context + ['SEP']
            entity_label = d['entity_label']
            start_position = d['start_position']
            end_position = d['end_position']
            examples.append((query, context, text, entity_label, start_position, end_position))
    for i in range(0,len(examples),batch_size):
        batch = examples[i:i+batch_size]
        texts_ids = []
        start_positions = []
        end_positions = []
        for _,_,text,_,start_position,end_position in batch:
            text_ids = tokenizer.convert_tokens_to_ids(text)
            texts_ids.append(text_ids)
            start_positions.append(start_position)
            end_positions.append(end_position)
        batch_texts = pad_sequence([torch.tensor(x) for x in texts_ids])
        attention_mask = torch.ones(batch_texts.shape, dtype=torch.uint8)
        for i in range(batch_texts.shape[0]):
            for j in range(batch_texts.shape[1]):
                if j+1>len(texts_ids[i]):
                    attention_mask[i][j] = 0

        yield batch_texts, attention_mask

def collate_fn(batch):
    nbatch = {}
    for d in batch:
        for k,v in d.items():
            nbatch[k]=nbatch.get(k,[])+[v]
    return nbatch


def collate_fn2(batch):
    nbatch = {}
    for d in batch:
        for k,v in d.items():
            nbatch[k]=nbatch.get(k,[])+[v]
    text = nbatch['text']
    start = pad_sequence(nbatch['start'],batch_first=True)
    end = pad_sequence(nbatch['end'],batch_first=True)
    segment_id = pad_sequence(nbatch['segment_id'],batch_first=True)
    max_len = start.shape[0]
    mask = torch.zeros(start.shape)
    for i in range(start.shape[0]):
        text_len = len(text[i])
        mask[:text_len+1]=1
    text = pad_sequence(nbatch['text'],batch_first=True)
    nbatch['text']=text
    nbatch['start']=start
    nbatch['end']=end
    nbatch['segment_id']=segment_id
    nbatch['mask']=mask
    return nbatch

class MyDataset2:
    def __init__(self,path, tokenizer):
        """
        Args:
            path: 待处理的文件路径
        """
        with open(path,encoding='utf-8') as f:
            self.data = json.load(f)
        self.examples = []
        self.texts = []
        self.segment_ids = []
        self.start_targets = []
        self.end_targets = []
        self.spans = []
        for d in self.data:
            if not d['impossible']:
                self.examples.append(d)
                context = d['context']
                context = context.split()
                query = d['query']
                query = list(query)
                #添加预训练模型的输入
                text = ['[CLS]']+query+['[SEP]']+context+['[SEP]']
                text = tokenizer.convert_tokens_to_ids(text)
                text = torch.tensor(text)
                self.texts.append(text)
                #添加segment id
                segment_id = torch.zeros(text.shape)
                segment_id[len(query)+1:]=1
                self.segment_ids.append(segment_id)
                #添加start_target
                start_position = d['start_position']
                start_target = torch.zeros(text.shape)
                for s in start_position:
                    start_target[s+len(query)+1] = 1
                self.start_targets.append(start_target)
                #添加end_target
                end_position = d['end_position']
                end_target = torch.zeros(text.shape)
                for e in end_position:
                    end_target[e+len(query)+1]=1
                self.end_targets.append(end_target)
                span = list(zip(start_position,end_position))
                self.spans.append(span)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {"example":self.examples[i],'text':self.texts[i],
                'start':self.start_targets[i],'end':self.end_targets[i],
                'span':self.spans[i],'segment_id':self.segment_ids[i]}


class MyDataset(Dataset):
    def __init__(self,path, tokenizer):
        """
        Args:
            path: 待处理的文件路径
        """
        with open(path,encoding='utf-8') as f:
            self.data = json.load(f)
        self.examples = []
        self.texts = []
        self.masks = []
        self.start_targets = []
        self.end_targets = []
        self.spans = []
        for d in self.data:
            if not d['impossible']:
                self.examples.append(d)
                context = d['context']
                context = context.split()
                query = d['query']
                query = list(query)
                #添加预训练模型的输入
                text = ['[CLS]']+query+['[SEP]']+context+['[SEP]']
                pad_len = 512-len(text)
                text = text+['[PAD]']*pad_len
                text = tokenizer.convert_tokens_to_ids(text)
                self.texts.append(text)
                #添加mask
                mask = torch.zeros(512)
                #mask[1:len(query)+1] = 1
                #mask[len(query)+2:len(text)] = 1
                mask[:len(text)]=1
                self.masks.append(mask)
                start_position = d['start_position']
                #添加start_target
                start_target = torch.zeros(512)
                for s in start_position:
                    start_target[s+len(query)+1] = 1
                self.start_targets.append(start_target)
                #添加end_target
                end_position = d['end_position']
                end_target = torch.zeros(512)
                for e in end_position:
                    end_target[e+len(query)+1]=1
                self.end_targets.append(end_target)
                span = list(zip(start_position,end_position))
                self.spans.append(span)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {"example":self.examples[i],'text':self.texts[i],
                'start':self.start_targets[i],'end':self.end_targets[i],
                'span':self.spans[i],'mask':self.masks[i]}
