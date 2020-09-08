import os
import time 

#cmd = "python -m torch.distributed.launch --nproc_per_node 4 train.py"
cmd = "CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node 3 train.py"
#cmd = "CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 train.py"
cmd1 = cmd + " --max_tokens 600"#似乎max_tokens对结果的影响还比较大
cmd2 = cmd + " --lr 2e-5"
cmd3 = cmd + " --pretrained_model_name_or_path /home/wangnan/mrc4ner/pretrained_models/RoBERTa_zh_Large_PyTorch/"#不知道是不是错觉，感觉这个模型上训练要慢一些
cmd4 = cmd + " --cls"#这个该有bug,加了cls为啥越跌代loss越大，心态崩了，然后几个个epoch直接score全为0，然后这个cls loss一直**居高不下**,span loss也一直无法收敛！
cmd5 = cmd + " --reduction sum"
cmd6 = cmd + " --allow_impossible"#如果不结合full,会出现span没有损失的情况，心酸，loss都到1e-05这个级别了， f1才0.57
cmd7 = cmd + " --warmup_ratio 0.1"
cmd8 = cmd + " --train_span_method full"#full但是not allow impossible，和cmd6有一些区别
cmd9 = cmd +" --allow_impossible"
cmd10 = cmd +" --dropout_prob 0.2"
cmd11 = cmd +" --span_layer"
cmd12 = cmd +" --cls"
cmd13 = cmd +" --cls  --span_layer"
cmd14 = cmd +"  --cls --span_layer --allow_impossible"#这个loss变化几乎让人绝望。。。为啥连start和end都能给你弄出特别大的损失。。。
cmd15 = cmd +"  --dropout_prob 0.2 --span_layer  --allow_impossible --warmup_ratio -1 --lr 8e-6  --train_span_method full"#这里必须改max_tokens，不然内存不够
cmd16 = cmd14 + ' --lr 8e-6'
cmd17 = cmd14 + " --lr 5e-5"
cmd18 = cmd14 + " --max_tokens 450"
cmd19 = cmd14 + " --train_span_method full"
cmd20 = cmd14 + "  --lr 8e-6  --max_tokens 450"#注意，这条命令在2个gpu和3个gpu上进行试验验证GPU数量对试验结果的影响，模型参数上看不出不同
cmd21 = cmd20 + " --warmup_ratio -1"
cmd22 = cmd + " --cls --allow_impossible --lr 8e-6  --max_tokens 450"
cmd23 = cmd +"  --dropout_prob 0.2 --span_layer  --allow_impossible --warmup_ratio -1 --lr 8e-6"#这个才是真正的作者的模型
cmd24 = cmd +"  --cls  --dropout_prob 0.2   --lr 8e-6  --max_tokens 450  --pretrained_model_name_or_path /home/wangnan/mrc4ner/pretrained_models/RoBERTa_zh_Large_PyTorch/  --allow_impossible"
cmd25 = cmd20 + "  --pretrained_model_name_or_path /home/wangnan/mrc4ner/pretrained_models/RoBERTa_zh_Large_PyTorch/"
cmd26 = cmd20 + "  --train_span_method  gold"
cmd27 = cmd26 + "  --max_grad_norm 1"
cmd28 = cmd23 + "  --max_grad_norm 1    --train_span_method  gold"#这个是作者的模型
cmd29 = cmd26 + "  --max_grad_norm 0.5"
cmd30 = cmd27.replace("--cls"," ")
cmd31 = cmd26.replace("450","512") +"  --max_grad_norm  0.75"
cmd32 = cmd31.replace("--cls"," ")
cmd33 = (cmd26+" --max_grad_norm 0.5").replace("8e-6","1e-5")
def run(cmd):
    print(cmd)
    log_file  = "/home/wangnan/mrc4ner/log/bash_log/%s.txt"%_id
    cmd = cmd+ ">>"
    f = os.popen(cmd,'r')
    output = f.read()
    _id = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    with open("/home/wangnan/mrc4ner/log/bash_log/%s.txt"%_id,'w') as f1:
        output = cmd+"\n\n"+output
        f1.write(output)
if __name__=="__main__":
    test_cmd = " --not_store --train_path ./datasets/OntoNotes4.0/testdata.json --dev_path ./datasets/OntoNotes4.0/testdata.json"
    #commands = [cmd12,cmd13,cmd14]
    #commands = [cmd16,cmd17,cmd18,cmd19]
    commands = [cmd33]
    for command in commands:
        print(command)
        os.system(command)
    '''
    commands = [cmd4,cmd5,cmd6]
    for command in commands:
        print(command)
        try:
            os.system(command)
        except:
            print("EXCEPTION!")
    '''