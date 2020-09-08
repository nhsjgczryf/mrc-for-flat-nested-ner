import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertTokenizer




def collate_fn(batch):
    nbatch = {}
    for d in batch:
        for k,v in d.items():
            nbatch[k]=nbatch.get(k,[])+[v]
    text = nbatch['text']
    n_start = pad_sequence(nbatch['start'],batch_first=True)
    n_end = pad_sequence(nbatch['end'],batch_first=True)
    segment_id = pad_sequence(nbatch['segment_id'],batch_first=True)
    max_len = n_start.shape[0]
    #这个mask对batch数据有用
    mask = torch.zeros(n_start.shape)
    for i in range(n_start.shape[0]):
        text_len = len(text[i])
        mask[:text_len+1]=1
    text = pad_sequence(nbatch['text'],batch_first=True)
    nbatch['text']=text
    nbatch['start']=n_start
    nbatch['end']=n_end
    nbatch['segment_id']=segment_id
    nbatch['mask']=mask
    #处理spans
    span = nbatch['span']
    max_span_len = max([len(s) for s in span])
    span_tensor = torch.full([len(text),max_span_len*2],-1,dtype=torch.long)
    for i in range(len(text)):
        for j,(s,e )in enumerate(span[i]):
            span_tensor[i][2*j]=s
            span_tensor[i][2*j+1]=e
    nbatch['span_tensor']=span_tensor
    return nbatch

def dist_collate_fn(batch):
    #dataset已经进行batch打包了，这里不需要再打包,去掉多余的维度
    return  batch[0]

class MyDataset:
    def __init__(self,path, tokenizer, allow_impossible=True):
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
            if (not allow_impossible) and d['impossible']:
                continue
            else:
                context = d['context']
                context = context.split()
                start_position = d['start_position']
                end_position = d['end_position']
                query = d['query']
                query,context,start_position,end_position = trans(tokenizer,query,context,start_position,end_position)
                #添加预训练模型的输入
                text = ['[CLS]']+query+['[SEP]']+context+['[SEP]']
                text = tokenizer.convert_tokens_to_ids(text)
                text = torch.tensor(text)
                self.texts.append(text)
                #添加segment id
                segment_id = torch.zeros(text.shape)
                segment_id[len(query)+2:]=1
                self.segment_ids.append(segment_id)
                #添加start_target,end_target,span
                start_target = torch.zeros(text.shape)
                end_target = torch.zeros(text.shape)
                span = []
                span = []
                start_target = torch.zeros(text.shape)
                end_target = torch.zeros(text.shape)
                for s,e in zip(start_position,end_position):
                    start_target[s] = 1
                    end_target[e] = 1
                    span.append((s,e))
                self.start_targets.append(start_target)
                self.end_targets.append(end_target)
                self.spans.append(span)
                self.examples.append(d)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {"example":self.examples[i],'text':self.texts[i],
                'start':self.start_targets[i],'end':self.end_targets[i],
                'span':self.spans[i],'segment_id':self.segment_ids[i]}

def trans(tokenizer,query,context,start_position,end_position,max_len=200):
    """
    输入中的query是str的字符串和context是list
    """
    query1 = tokenizer.tokenize(query)
    context1 = []
    for word in context:
        word1 = tokenizer.tokenize(word)
        context1.append(word1)
    context2 = []
    for c in context1:
        context2.extend(c)
    start_position1 = []
    end_position1 = []
    for s,e in zip(start_position,end_position):
        s1 = sum([len(c) for c in context1[:s]])
        e1 = sum([len(c) for c in context1[:e]])
        s2 = len(query)+2+s1
        e2 = len(query)+2+e1
        if s2>=max_len-1 or e2>=max_len-1:
            continue
        start_position1.append(s2)
        end_position1.append(e2)
    return query1, context2,start_position1,end_position1
    
    
class BatchDataset:
    """
    这个是按照max_token这个参数对进行打包后的数据
    """
    def __init__(self, path, tokenizer,max_tokens, allow_impossible=False):
        with open(path, encoding='utf-8') as f:
            self.data = json.load(f)
        self.texts = []
        self.segment_ids = []
        self.start_targets = []
        self.end_targets = []
        self.spans = []
        self.max_tokens = max_tokens
        for d in self.data:
            if (not allow_impossible) and d['impossible']:
                continue
            else:
                context = d['context']
                context = context.split()
                start_position = d['start_position']
                end_position = d['end_position']
                query = d['query']
                query,context,start_position,end_position = trans(tokenizer,query,context,start_position,end_position)
                # 添加预训练模型的输入,这里设置去文本长度最大为200
                text = ['[CLS]'] + query + ['[SEP]'] + context[:200] + ['[SEP]']
                text = tokenizer.convert_tokens_to_ids(text)
                text = torch.tensor(text)
                self.texts.append(text)
                # 添加segment id
                segment_id = torch.zeros(text.shape)
                segment_id[len(query) + 2:] = 1
                self.segment_ids.append(segment_id)
                # 添加start_target,end_target,span
                span = []
                start_target = torch.zeros(text.shape)
                end_target = torch.zeros(text.shape)
                for s,e in zip(start_position,end_position):
                    start_target[s] = 1
                    end_target[e] = 1
                    span.append((s,e))
                self.start_targets.append(start_target)
                self.end_targets.append(end_target)
                self.spans.append(span)
        #下面进行处理
        t = zip(self.texts,self.segment_ids,self.start_targets,self.end_targets,self.spans)
        self.texts,self.segment_ids,self.start_targets,self.end_targets,self.spans = zip(*sorted(t,key = lambda x:len(x[0])))
        i=0
        indexs = []#记录每个batch的start和end对应的index
        #下面这个循环有两个bug，第一个是如果某个样本的长度超过了batch允许的max_len会出错。
        #第二个是可能存在最后一个样本丢失的情况（当只剩最后一个样本时，应该单独作为batch，但是会被舍弃）
        #其实就是无法容纳一个样本作为batch的情况
        while i<len(self.texts):
            #print("{}/{}\n".format(i,len(self.texts)))
            batch_tokens = 0
            batch_max_len = 0
            batch = []
            j=i
            while batch_tokens<=max_tokens:
                if j>=len(self.texts):
                    break
                batch.append(self.texts[j])
                batch_max_len = max(batch_max_len,len(self.texts[j]))
                batch_tokens = batch_max_len*len(batch)
                j+=1
            #上面循环退出的时候，batch_tokens>max_tokens
            if j>=len(self.texts):
                break
            batch = batch[:-1]
            indexs.append((i,i+len(batch)))
            i = j-1
        self.texts1 = []
        self.segment_ids1 = []
        self.start_targets1 = []
        self.end_targets1 = []
        self.span_tensor = []
        self.mask = []
        for s,e in indexs:
            txt = self.texts[s:e]
            text = txt
            seg =self.segment_ids[s:e]
            st = self.start_targets[s:e]
            et = self.end_targets[s:e]
            txt = pad_sequence(txt,batch_first=True)
            seg = pad_sequence(seg,batch_first=True)
            st = pad_sequence(st,batch_first=True)
            et = pad_sequence(et,batch_first=True)
            span = self.spans[s:e]
            max_span_len = max([len(s) for s in span])
            ste = torch.full([len(text),max_span_len*2],-1,dtype=torch.long)
            for i in range(len(text)):
                for j, (s, e) in enumerate(span[i]):
                    try:
                        ste[i][2 * j] = s
                        ste[i][2 * j + 1] = e
                    except:
                        print(span[i],j,s,e,ste[i])
            mask = torch.zeros(st.shape)
            for i in range(st.shape[0]):
                text_len = len(text[i])
                mask[:text_len + 1] = 1
            self.texts1.append(txt)
            self.segment_ids1.append(seg)
            self.start_targets1.append(st)
            self.end_targets1.append(et)
            self.span_tensor.append(ste)
            self.mask.append(mask)


    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, i):
        #print(i,len(self.start_targets1))
        return {'text': self.texts1[i],'segment_id':self.segment_ids1[i],
                'start': self.start_targets1[i], 'end': self.end_targets1[i],
                'span_tensor': self.span_tensor[i],'mask':self.mask[i]}

def dist_load_data(pretrained_model_name_or_path, path,max_tokens,allow_impossible=False):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    dataset = BatchDataset(path, tokenizer,max_tokens,allow_impossible)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dist_collate_fn,
                                  pin_memory=False, sampler=sampler)
    return dataloader

def load_data(pretrained_model_name_or_path,path, batch_size, shuffle=False,allow_impossible=False):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    dataset = MyDataset(path, tokenizer,allow_impossible)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  pin_memory=False,shuffle=shuffle)
    return dataloader
