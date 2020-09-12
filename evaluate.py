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


def evaluation(model, dataloader, train_eval=True):
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
    tqdm_dataloader = tqdm(dataloader,desc='eval',ncols=100)
    with torch.no_grad():
        for i,batch in enumerate(tqdm_dataloader):
            #print(i)
            text, mask, segment_id, gold_spans = batch['text'], batch['mask'], batch['segment_id'],batch['span']
            text, mask, segment_id = text.to(device),mask.to(device), segment_id.to(device)
            try:
                pre_spans = model.predict(text,mask,segment_id,mask_decode)
                #for d in batch['example']:
                #    interaction(model,tokenizer,d)
                gold.append((i,gold_spans))
                predict.append((i,pre_spans))
            except Exception as e:
                print(text, mask, segment_id)
                print(e)
                print(i)
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

def test(model_dir,file,batch_size,args,tensorboard_writer):
    path = model_dir+file
    cpt = torch.load(path)
    mymodel = MyModel(args)
    mymodel.load_state_dict(cpt["state_dict"])
    test_dataset = pre1_test_dataset if args.pretrained_model_name_or_path==pretrained_model1 else pre2_test_dataset
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn,
                              pin_memory=False,shuffle=False)
    print(path,'\n',args)
    time.sleep(0.1)
    p,r,f = evaluation(mymodel,test_dataloader,train_eval=False)
    print("model {} test dataset:\nprecision:{:.4f} recall:{:.4f} f1:{:.4f}".format(file,p,r,f))
    tensorboard_writer.add_scalars("score_%s"%file,{'precision':p,'recall':r,'f1':f})
    tensorboard_writer.flush()
    return p,r,f