from torch import nn
import torch
from torch.nn import functional as F


class Normalized_Focal_Loss(nn.Module):
  def __init__(self,alpha=0.1,gamma=2,weight_grad=False,reduction='norm'):
    """alpha=p*gamma/(1-p)，p为梯度急速下降的拐点，，当gamma=2的时候,p=0.05的时候,alpha=0.1"""
    super(Normalized_Focal_Loss,self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.weight_grad = weight_grad
    self.reduction=reduction

  def forward(self,predict,target):
    """
    predict: (N,C) or (N) in binary case
    target: (N)
    return:
    """
    #print(predict.shape,target.shape)
    target = target.view(-1,1)
    target = target.long()
    predict = predict.squeeze()
    if len(predict.shape)==0:
      predict = predict.view(-1)
    if len(predict.shape)==1:
      """
      先计算p会导致log(p)的计算可能出现nan
      simoid(x) = 1/（1+exp(-x)） = exp(x/2)/(exp(x/2)+exp(-x/2))
      1-sigmoid(x) = sigmoid(-x)
      转化为softmax的形式来计算
      """
      predict = torch.stack([-predict/2,predict/2],dim=1)
    predict_logit = F.log_softmax(predict,dim=-1) #（N,C）
    predict_prob = predict_logit.detach().exp() if not self.weight_grad else predict_logit.exp()#(N)
    predict_logit = predict_logit.gather(1,target).view(-1) #(N)
    predict_prob = predict_prob.gather(1,target).view(-1) #(N)
    if self.reduction.startswith("n_") or self.reduction=='norm':
      weight = predict_prob**self.alpha*(1-predict_prob)**self.gamma
      if self.reduction=='norm':
        weight = weight + 1e-12#防止除零导致的nan
        weight = weight/weight.sum()
      elif self.reduction=="n_mean":
        weight = weight/len(weight)
    elif self.reduction == "mean":
      weight = torch.full(predict_prob.shape,1/len(predict_prob)).type_as(predict_prob)
    elif self.reduction == "sum":
      weight = torch.full(predict_prob.shape,1.0).type_as(predict_prob)
    if not self.reduction.endswith("none"):
      loss = -(weight*predict_logit).sum()
      #print("weight:",weight)
      #print("weight_sum",weight.sum())
      #print("max loss",torch.max(-predict_logit))
      #print("loss:",loss.item())
      #print(weight)
    else:
      loss = predict_logit
    return loss
  
class DynamicLoss(nn.Module):
  def __init__(self,num_labels,reduction='mean',beta=0.9,upper_bound=0.1,hard_weight=True):
    """
    Args:
      beta: 指数平滑的系数
      upper_bound: 这个参数会在我们一个batch里面全部预测正确的时候影响我们梯度的大小，所以可能对不同batch之间的类别不平衡有影响。(但是必须在mean的情况下才能体现)
      hard_weight: 是否估计困难样本的概率阈值(为True会增加loss的计算量)
    """
    super(DynamicLoss,self).__init__()
    self.reduction = reduction
    #这个是我们正确分类样本的的lambda*(1-p)中的lambda
    self.beta=beta
    self.upper_bound = upper_bound
    self.hard_weight = hard_weight
    self.class_prob = torch.full([num_labels],1/num_labels)

  def forward(self,predict,target):
    """
    predict: (N,C) or (N) in binary case
    target: (N)
    return:
    """
    self.class_prob = self.class_prob.type_as(predict)
    target = target.view(-1)
    target = target.long()
    predict = predict.squeeze()
    if len(predict.shape)==0:
      predict = predict.view(-1)
    if len(predict.shape)==1:
      """
      先计算p会导致log(p)的计算可能出现nan
      simoid(x) = 1/（1+exp(-x)） = exp(x/2)/(exp(x/2)+exp(-x/2))
      1-sigmoid(x) = sigmoid(-x)
      转化为softmax的形式来计算
      """
      predict = torch.stack([-predict/2,predict/2],dim=1)
    predict_logit = F.log_softmax(predict,dim=-1) #（N,C）
    predict_prob = predict_logit.detach().clone().exp() #(N)
    predict_idx = predict_logit.argmax(dim=-1)
  
    correct = torch.eq(predict_idx,target.view(-1))
    wrong = torch.logical_not(correct)
    wrong_prob = predict_prob[wrong]
    
    #计算正确分类样本的权重
    wrong_prob_target = wrong_prob.gather(1,target[wrong].view(-1,1))
    if len(wrong_prob_target)!=0:
      wrong_prob_min = wrong_prob_target.min() #错误分类样本中，最小的概率
      correct_weight = min(self.upper_bound,wrong_prob_min/correct.sum())
    else:
      #如果我们一个batch里面预测的结果全部正确
      correct_weight = self.upper_bound

    weight = torch.zeros(target.shape).type_as(predict)
    weight[correct]=correct_weight
    if self.hard_weight:
      wrong_prob_predict = wrong_prob.gather(1,predict_idx[wrong].view(-1,1))
      class_prob_padded = torch.stack([self.class_prob for i in range(len(wrong))],dim=0)
      class_prob_predict = class_prob_padded.gather(1,predict_idx[wrong].view(-1,1))
      #统计一下当前batch对正确分类样本的概率
      correct_col_idx = predict_idx[correct]
      correct_row_idx = torch.nonzero(correct.int(),as_tuple=True)[0]
      correct_mask = torch.zeros(predict_prob.shape).type_as(predict_prob)
      correct_mask[correct_row_idx,correct_col_idx]=1
      masked_predict_prob = correct_mask*predict_prob
      #注意考虑某个类别没有出现在batch中的情况，会出现inf
      average_prob = masked_predict_prob.sum(dim=0)/(correct_mask.sum(dim=0))
      class_mask = (correct_mask.sum(dim=0)!=0)
      #更新各个类别的预测概率
      self.class_prob[class_mask]=self.beta*average_prob[class_mask]+(1-self.beta)*self.class_prob[class_mask]
      
      wrong_hard = torch.gt(wrong_prob_predict,class_prob_predict)
      wrong_easy = torch.logical_not(wrong_hard)
      wrong_hard_prob = wrong_prob_target[wrong_hard]
      wrong_hard_cls = class_prob_predict[wrong_hard]
      wrong_hard_weight = wrong_hard_prob*wrong_hard_cls/((1-wrong_hard_prob)*(1-wrong_hard_cls))
      wrong_hard_mask = wrong.clone()
      wrong_easy_mask = wrong.clone()
      wrong_hard_mask[wrong]=wrong_hard.view(-1)
      wrong_easy_mask[wrong]=wrong_easy.view(-1)
      weight[wrong_hard_mask]=wrong_hard_weight
      weight[wrong_easy_mask]=1 #我们对easy的样本的梯度不做缩小处理改变
    else:
      weight[wrong]=1
    #print(weight)
    #print(weight.sum())
    predict_logit_target = predict_logit.gather(1,target.view(-1,1))
    loss = (weight*predict_logit_target).sum()
    if self.reduction=='mean':
      loss =  loss/len(weight)
    elif self.reduction=='norm':
      loss = loss/(weight.sum())
    return -loss