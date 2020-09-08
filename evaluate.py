import torch
from mydataset import load_data,trans
from tqdm import tqdm
import math
from transformers import BertTokenizer

def get_score(gold_set,predict_set):
    """得到两个集合的precision,recall.f1"""
    TP = len(set.intersection(gold_set,predict_set))
    precision = TP/(len(predict_set)+1e-6)
    recall = TP/(len(gold_set)+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return precision,recall,f1

def dev_test_eval(model,pretrained_model_name_or_path,dev_path,test_path):
    """在开发集和测试集上进行评估"""
    dev_dataloader = load_data(pretrained_model_name_or_path,dev_path, 10, 
                               shuffle=False,allow_impossible=True)
    test_dataloader = load_data(pretrained_model_name_or_path,test_path, 10, 
                                shuffle=False,allow_impossible=True)
    dev_precision,dev_recall,dev_f1 = evaluation(model, dev_dataloader)
    test_precision,test_recall,dev_f1 = evaluation(model, test_dataloader)
    print("dev_precision:",dev_precision,"dev_recall",dev_recall,"dev_f1",dev_f1)
    print("test_precision:",dev_precision,"test_recall",dev_recall,"test_f1",dev_f1)


def evaluation(model, dataloader, train_eval=True,desc=None):
    """这部分代码用于训练过程中的评估"""
    if hasattr(model,'module'):
        model = model.module
    model.eval()
    mask_decode = (not train_eval)
    #tokenizer = BertTokenizer.from_pretrained(model.config.pretrained_model_name_or_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    #print(device)
    gold = []
    predict = []
    tqdm_dataloader = tqdm(dataloader,desc=desc)
    with torch.no_grad():
        for i,batch in enumerate(tqdm_dataloader):
            text, mask, segment_id, gold_spans = batch['text'], batch['mask'], batch['segment_id'],batch['span']
            text, mask, segment_id = text.to(device),mask.to(device), segment_id.to(device)
            pre_spans = model.predict(text,mask,segment_id,mask_decode)
            #for d in batch['example']:
            #    interaction(model,tokenizer,d)
            gold.append((i,gold_spans))
            predict.append((i,pre_spans))
    gold2 = set()
    predict2 = set()
    for g in gold:
        i,gold_spans = g
        for j,gs in enumerate(gold_spans):
            for gsi in gs:
                item = (i,j,gsi[0],gsi[1])
                gold2.add(item)
    for p in predict:
        i,pre_spans = p
        for j, ps in enumerate(pre_spans):
            for psi in ps:
                item = (i,j, psi[0],psi[1])
                predict2.add(item)
    precision,recall,f1 = get_score(gold2,predict2)
    return precision,recall,f1


def interaction(model,tokenizer,d):
    #只对一个样本处理
    model.eval()
    context = d['context']
    #print(d)
    context = context.split()
    start_position = d['start_position']
    end_position = d['end_position']
    query = d['query']
    query,context,start_position,end_position = trans(tokenizer,query,context,start_position,end_position)
    text = ["[CLS]"]+query+["[SEP]"]+context+["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(text)
    input_ids = torch.tensor(input_ids)
    mask = torch.zeros(input_ids.shape)
    mask[:len(text)+1]=1
    mask = mask.view(1,-1)
    segment_id = torch.zeros(input_ids.shape)
    segment_id[len(query)+2:] = 1
    input_ids = input_ids.view(1,-1)
    segment_id = segment_id.view(1,-1)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        mask = mask.cuda()
        segment_id = segment_id.cuda()
        model = model.cuda()
    spans = model.predict(input_ids,mask,segment_id)
    #print("with mask decode")
    #print(spans)
    #print("without mask decode")
    #print(model.predict(input_ids,mask,segment_id,False))
    #下面是计算loss
    '''
    start_target = torch.zeros(len(text))
    end_target = torch.zeros(len(text))
    span_tensor = torch.full((len(text)*2,),-1,dtype=torch.long)
    for i,(s,e) in enumerate(zip(start_position,end_position)):
        start_target[s] = 1
        end_target[e] = 1
        span_tensor[2*i]=s
        span_tensor[2*i+1]=e
    loss = model(input_ids,mask.cuda(),segment_id.cuda(),start_target.view(1,-1).cuda(),
                 end_target.view(1,-1).cuda(), span_tensor.view(1,-1).cuda())
    '''
    spans = spans[0]
    text_spans = []
    for s,e in spans:
        span = text[s:e+1]
        span = tokenizer.convert_tokens_to_string(span)
        text_spans.append(span)
    gold_span = []
    for s,e in zip(start_position,end_position):
        span = text[s:e+1]
        span = tokenizer.convert_tokens_to_string(span)
        gold_span.append(span)
    text = tokenizer.convert_tokens_to_string(text)
    print("text:",text)
    print("predict_span_idx:",spans)
    print("gold_span_idx:",list(zip(start_position,end_position)))
    print("predict_text_span:",text_spans)
    print("gold_text_span:",gold_span)
    return text_spans