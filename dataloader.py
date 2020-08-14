"""
这个文件的作用是
把数据转换为batch的feature
"""
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from  transformers import *

tokenizer = None
labels = {'LOC':0,'PER':1,'GPE':2,'ORG':3}
#导入文件
path = r'C:\Users\DELL\Desktop\mrc4nner\datasets\OntoNotes4.0\mrc-ner.train'
with open(path,encoding='utf-8') as f:
    data = json.load(f)
#提取数据
examples = []
for d in data:
    if not d['impossible']:
        context = d['context']
        context = context.split()
        query = d['query']
        query = list(query)
        text = ['CLS']+query+['SEP']+context+['SEP']
        entity_label = d['entity_label']
        start_position  = d['start_position']
        end_position = d['end_position']
        examples.append((query,context,text,entity_label,start_position,end_position))

def get_batch_data(batch_size):
    """
    :param batch_size:
    :return:
    """
    for i in range(0,len(examples),batch_size):
        batch = examples[i:i+batch_size]
        texts_ids = []
        start_positions = []
        end_positions = []
        start_targets = []
        end_targets = []
        spans = []
        for _,_,text,_,start_position,end_position in batch:
            text_ids = tokenizer.convert_tokens_to_ids(text)
            texts_ids.append(text_ids)
            spans.append((start_position,end_position))
            start_positions.append(start_position)
            end_positions.append(end_position)
        batch_texts = pad_sequence([torch.tensor(x) for x in texts_ids])
        attention_mask = torch.ones(batch_texts.shape, dtype=torch.uint8)
        for i in range(batch_texts.shape[0]):
            for j in range(batch_texts.shape[1]):
                if j+1>len(texts_ids[i]):
                    attention_mask[i][j] = 0

        yield batch_texts, attention_mask, start_targets, end_targets, spans
