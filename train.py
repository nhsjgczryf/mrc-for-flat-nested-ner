'''
这个文件的功能是：
训练模型
保存训练好的模型
保存训练过程中的历史纪录

'''
import time
from model import MyModel
import dataloader
import torch
import config

model = MyModel(config)
model.train()
optimizer = torch.optim.adamax(model.parameters())
for e in range(len(config.epoches)):
    for batch in dataloader.get_batch_data(config.batch_size):
        model.zero_grad()
        input, mask, start_target, end_target, spans = batch
        loss = model.loss(input,mask,start_target,end_target,spans)
        loss.backward()
        optimizer.step()
