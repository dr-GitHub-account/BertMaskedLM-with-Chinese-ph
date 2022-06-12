#!/usr/bin/env python
# coding=utf-8
"""
   Copyright (C) 2019 Tuya NLP. All rights reserved.   
   FileName：train.py
   Author  ：shizi
   DateTime：2021/7/14
   Desc    ：该脚本只支持bert和gpt2的训练。经过代码抽象后，要求当修改model_name=bert时，
             能够自动调用bert分词，训练Bert模型；当修改model_name=gpt2时，能够自动调用
             gpt分词，训练gpt模型.
             目前不支持char模式。
   Vocab   : SougouBertVocab共68181个词汇，它们是过滤了Sougou语料词频不大于6*1e-7的词
             后，与bert-base-chinese自带的vocab取并集
"""

import torch
import logging
import argparse
from transformers import AdamW
import torch.utils.data as Data
from transformers import GPT2Config,BertConfig
from transformers import GPT2LMHeadModel,BertForMaskedLM
from utils import (calculate_loss_and_accuracy,GptTokenTool,BertTokenTool,
                   MyDataset,day_month,ModelWrapper,bertshow_predict_vs_actual)
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import time
import pdb
import numpy as np

# in original env, setuptools         61.2.0

PAD = '[PAD]'
pad_id = 0
logger = None

# 命令行参数
def parse_args():
    """
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str,default='bert', 
                        help='model name = gpt2 or bert')
    parser.add_argument('--trainfile', type=str, 
                        default="./testdata/stard1w.txt", 
                        help='training data path')
    parser.add_argument('--savemodel', type=str, 
                        default='./modelfile/MN/', 
                        help='model file save path')
    parser.add_argument('--mode',type=str,
                        default='word',
                        help='train model based on word or char')
    parser.add_argument('--corpusname',type=str,
                        default='sougou',
                        help='trainging corpus name')
    parser.add_argument('--vocabpath', type=str, 
                        default='./vocab/SougouBertVocab.txt', 
                        help='vocabulary file')
    parser.add_argument('--batchsize', type=int, default=2, 
                        help='set your batch_size')
    parser.add_argument('--epoch', type=int, default=1, 
                        help='epoch')
    parser.add_argument('--showstep', type=int, default=100, 
                        help='''during training, save and show model after 
                        train N steps ''')
    parser.add_argument('--usegpu', type=int,default=0, 
                        help="usegpu = 1 if use else 0")
    parser.add_argument('--device', type=str, default='0',
                        help="""if usegpu and only a sigle gpu, you can ignore 
                        the item, but if you use multi gpu, you should set the 
                        item='0,1' for two pieces of gpu, item='0,1,2' for 
                        three pieces of gpu and so on""")
    parser.add_argument('--loadmodel', type=str, default=None,
                        help="your trained model file address")
    parser.add_argument('--log', type=str, default='./log/MN/CN_DT_train_log.txt')
    parser.add_argument('--tensorboard',type=str,default='./tensorboard/MN/CN_DT_train',
                        help='tensorboard directory')
    parser.add_argument('--curepoch',type=str, default=None, 
                        help="what epoch you want to begin train model")
    parser.add_argument('--curstep',type=str, default=None, 
                        help="where step you want to begin train model")
    
    parser.add_argument('--outputinfo',type=str, default=None, 
                        help="output time and num of repeat")
    
    args = parser.parse_args()
    return args

# get arguments
args = parse_args()
# 训练数据 按行处理
assert args.modelname in ['gpt2','bert'], 'model_name must be gpt2 or bert'
assert args.usegpu in [0,1],'usegpu is equal 0 or 1'
assert args.mode in ['word','char'], 'mode is word or char'

day = day_month(datetime.now())

# log相关
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# default='./log/MN/CN_DT_train_log.txt'
logfile = args.log
# args.trainfile is the training data file, default="./testdata/stard1w.txt"
corpusname = args.trainfile.split('/')[-1]
corpusname = corpusname.split('.')[0]
logfile = logfile.replace('MN',args.modelname)
# logfile = logfile.replace('CN',corpusname).replace('DT',day)
logfile = logfile.replace('CN',corpusname).replace('DT',args.outputinfo)
file_handler = logging.FileHandler(filename=logfile,mode='w')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(console)

# tensorboard相关
# args.tensorboard is the file corresponding to tensorboard, default='./tensorboard/MN/CN_DT_train'
# tb_dir = args.tensorboard.replace('MN',args.modelname)
# tb_dir = tb_dir.replace('CN',corpusname).replace('DT',day)
# if os.path.exists(tb_dir):
#     files = os.listdir(tb_dir)
#     for i in files:
#         os.remove(os.path.join(tb_dir,i))
# tb_writer = SummaryWriter(log_dir = tb_dir,filename_suffix='tb')

# 默认不进入下面的if
if args.modelname=='gpt2':
    configuration = GPT2Config(
                    vocab_size = 68181,
                    bos_token_id = 68180,
                    eos_token_id = 68180,
                    n_embd = 768 // 4,
                    n_layer= 12 // 4,
                    n_head= 12 // 4,
                    n_ctx = 1024//4,
                    n_positions = 1024 // 4)
    tokenizer = GptTokenTool(args.vocabpath)
    if not args.loadmodel:
        model = GPT2LMHeadModel(configuration)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.loadmodel)
# 默认进入下面的elif，用configuration或from_pretrained(args.loadmodel)进行BERT模型的实例化
elif args.modelname=='bert':
    # configuration = BertConfig(
    #                 vocab_size=68181,
    #                 hidden_size=768 // 4,
    #                 num_hidden_layers=12 // 4,
    #                 num_attention_heads=12 // 4,
    #                 intermediate_size=3072 // 4,
    #                 max_position_embeddings=512 // 4,
    #                 type_vocab_size=2,
    #                 pad_token_id=0,
    #                 return_dict=True)
    configuration = BertConfig(
                    vocab_size=21128,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    max_position_embeddings=512,
                    type_vocab_size=2,
                    pad_token_id=0,
                    return_dict=True)
    # 实例化BertTokenTool类对象tokenizer
    tokenizer = BertTokenTool(args.vocabpath)
    # 默认有args.loadmodel，不进入下面的if
    if not args.loadmodel:
        logger.info("*****Running model = BertForMaskedLM(configuration)*****")
        model = BertForMaskedLM(configuration)
    # 默认有args.loadmodel，进入下面的else
    else:
        logger.info("*****Running model = BertForMaskedLM.from_pretrained(args.loadmodel)*****")
        # from_pretrained返回加载了权重的model，至此，需训练的model被定义好了
        model = BertForMaskedLM.from_pretrained(args.loadmodel)
transfer = tokenizer.tokenizer.convert_ids_to_tokens

multi_gpu = False
# 默认进入下面的if，设置显卡相关
if bool(args.usegpu)==True and args.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert str(device)=='cuda','your machine need own a gpu card at least'
    udevice = list(map(int,args.device.split(',')))
    udevice = [i for i in udevice if type(i)==int]
    # 单卡则进入下面的if
    if len(udevice)==1:
        sdevice=torch.device(udevice[0])
        model = model.to(sdevice)
    # 多卡则进入下面的elif
    elif len(udevice)>1 and torch.cuda.device_count()>1:
        model = model.to(device)
        device_ids=[int(i) for i in udevice]
        model = torch.nn.DataParallel(model,device_ids=device_ids)
        multi_gpu = True
# 默认不进入下面的else
else:
    device = torch.device("cpu")

# args.trainfile is the training data file, default="./testdata/stard1w.txt"
trainset = MyDataset(args.trainfile, n_raws=1000, shuffle=True)

time0 = time.time()
# 优化器
optimizer = AdamW(model.parameters(), lr= 1e-5)

# 训练基础信息记录进日志
logger.info("The Initial Date = %s"%day)
logger.info("%s is training which based on corpus %s"%
            (args.modelname,args.trainfile))
logger.info("The log information is saved in : %s"%logfile)

# curepoch表示从第几个epoch开始，默认args.curepoch为None，curepoch为-1
curepoch = int(args.curepoch) if args.curepoch else -1
# curstep表示从第几个step开始，默认args.curepoch为None，curstep为-1
curstep = int(args.curstep) if args.curepoch else -1
    
#%%
# 此步骤会进行jieba分词与打乱顺序
# trainset在train()函数调用时，作为corpus参数被传入
trainset.initial()

# 训练函数
def train(model, 
          corpus, 
          epochs = args.epoch, 
          modelname = args.modelname, 
          batchs = args.batchsize,
          maxlength = 128):
    
    runloss = 0.
    runacc = 0.
    speci_var = 1
    
    # 遍历每一轮
    for ee in range(epochs):
        # 默认不进入下面的if，从中间的epoch开始训练时，才进入
        if ee < curepoch:
            continue
        # 实例化DataLoader类对象train_iter
        train_iter = Data.DataLoader(dataset=corpus, batch_size=batchs, shuffle=True)
        logger.info("epoch = %d"%ee)
        
        # 遍历一轮中的每个batch
        for gg, data in enumerate(train_iter):
            if gg < curstep:
                continue
            # 默认不进入下面的if，默认modelname为'bert'
            if modelname == 'gpt2':
                inputs,labels = tokenizer.tokenize(data, max_length=maxlength)
                if str(device)=='cuda':
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                outputs = model.forward(input_ids = inputs)
                loss, accuracy = calculate_loss_and_accuracy(outputs, labels=labels)
            # 默认进入下面的elif，默认modelname为'bert'
            elif modelname == 'bert':
                # tokenizer是BertTokenTool类对象
                # tokenizer.tokenize()定义：
                # def tokenize(self, word_list,  p_mask:float, max_length=300, truncation=True, 
                #              padding=True, islastone=False):
                #     # self.tokenizer是BertTokenizer类的实例，BertTokenizer类: Construct a BERT tokenizer. Based on WordPiece
                #     inputs = self.tokenizer(word_list,
                #                             return_tensors="pt",
                #                             truncation=truncation,
                #                             padding=padding,
                #                             max_length=max_length)
                #     # 默认p_mask = 0.15，不进入下面的if
                #     if p_mask == 0:
                #         if islastone:
                #             inputs['input_ids'],labels = self.collate_fn2(inputs['input_ids'])
                #         else:
                #             labels = None
                #     # 默认p_mask = 0.15，进入下面的else
                #     else:
                #         # collate_fn函数用来进一步校正数据，按照bert模型的mask原则作出masked的inputs与labels。
                #         # 其中，inputs有15%的几率被mask，而所有的mask中，有80%的几率用'[mask]'进行替代，有10%用随机词进行替代，剩下10%保留原形
                #         inputs['input_ids'],labels = self.collate_fn(inputs['input_ids'], p_mask=p_mask)
                #     return inputs,labels
                inputs,labels = tokenizer.tokenize(data, max_length=maxlength, p_mask=0.15)
                if ee == 0 and gg == 0:
                    logger.info("*****For ee == 0, gg == 0:*****")
                    logger.info("*****inputs: {}*****".format(inputs))
                    logger.info("*****np.shape(inputs): {}*****".format(np.shape(inputs)))
                    logger.info("*****np.shape(inputs['input_ids']): {}*****".format(np.shape(inputs['input_ids'])))
                    logger.info("*****np.shape(inputs['token_type_ids']): {}*****".format(np.shape(inputs['token_type_ids'])))
                    logger.info("*****np.shape(inputs['attention_mask']): {}*****".format(np.shape(inputs['attention_mask'])))
                    logger.info("*****labels: {}*****".format(labels))
                    logger.info("*****np.shape(labels): {}*****".format(np.shape(labels)))
                
                # 默认使用gpu，进入下面的if
                if str(device)=='cuda':
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                # 模型前向传播，返回MaskedLMOutput类对象outputs，outputs含有成员变量logits, loss
                outputs = model(**inputs, labels=labels)
                # np.shape(labels): torch.Size([batch_size, max_len])，其中被mask的token相应label不为-100
                # 得到的masked_label为一个一维张量，如tensor([  704, 14966])，其中每一个元素都是一个mask的label的id
                masked_label = labels[labels != -100]
                # mask的预测结果
                masked_pre = outputs.logits[labels != -100].max(-1).indices
                # masked_label.numel()为一维张量masked_label中包含元素的个数，即mask的个数
                # 如果masked_label.numel()为0则代表当前没有mask
                if masked_label.numel() == 0:
                    accuracy = 0
                # 如果masked_label.numel()不为0则代表当前有mask，能够计算出一个accuracy
                else:
                    accuracy = (torch.sum(masked_pre==masked_label)/masked_label.numel()).item()
                # 根据单卡还是多卡来进行处理outputs.loss，进而得到损失
                loss = outputs.loss.mean() if multi_gpu else outputs.loss
                # tensorboard相关
                if gg%(0.1*args.showstep)==9:
                    info2 = bertshow_predict_vs_actual(inputs, labels, outputs)
                    # tb_writer.add_text('predict-vs-actural',info2,ee*len(train_iter)+gg)
                    # tb_writer.close()
            # tensorboard相关
            if speci_var == 1:
                model_wr = ModelWrapper(model)
                if modelname == 'gpt2':
                    pass
                    # tb_writer.add_graph(model_wr, inputs)
                elif modelname == 'bert':
                    pass
                #     tb_writer.add_graph(model_wr, inputs['input_ids'])
                # tb_writer.close()
                speci_var = 0
            # 当前batch的loss累加到runloss上
            runloss += loss.item()
            # 当前batch的accuracy累加到runacc上
            runacc += accuracy
            optimizer.state.get("")
            if gg%(0.1*args.showstep)==9:
                time1 = time.time()
                logger.info('\t batch = %d \t loss = %.5f \t acc = %.3f \t cost_time = %.3fs'%(
                             gg,loss.item(),accuracy,time1-time0))
                # tb_writer.add_scalar('%s-train-loss'%args.modelname,runloss/0.1/args.showstep,ee*len(train_iter)+gg)
                # tb_writer.add_scalar('%s-train-acc'%args.modelname,runacc/0.1/args.showstep,ee*len(train_iter)+gg)
                runloss = 0.0
                runacc = 0.0
            # 梯度设置为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # **************************202206121708**************************
            if gg>100:
                break
            
            if gg % args.showstep == 0:
                save_path1 = os.path.join(args.savemodel.replace('MN',args.modelname),
                                          '%s_%s_%s_step_%d.bin'%(args.modelname,
                                                                  args.corpusname,args.mode,gg))
                if hasattr(model,'module'):
                    model.module.save_pretrained(save_path1)
                else:
                    model.save_pretrained(save_path1)
        save_path2 = os.path.join(args.savemodel.replace('MN',args.modelname),
                                  '%s_%s_%s_epoch_%d.bin'%(args.modelname,
                                                           args.corpusname,args.mode,ee))
        if hasattr(model,'module'):
            model.module.save_pretrained(save_path2)
        else:
            model.save_pretrained(save_path2)
        logger.info('we get model %s'%save_path2)
    # logger.info('tensorboard information has been recorded in %s'%tb_dir)
    logger.info('training done!')

train(model=model, corpus=trainset)
# tb_writer.close()




       
        
