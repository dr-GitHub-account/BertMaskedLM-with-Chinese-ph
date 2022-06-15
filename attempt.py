import torch
import jieba

# labels = torch.tensor([[ -100,  -100,  -100,  -100,  -100,  -100],
#         [ -100,  -100,  -100,  -100,  -100,  -100],
#         [ -100,  -100,  -100,  -100,  -100,  -100],
#         [ -100,  -100,  -100,  -100,  -100,  -100],
#         [ -100,   704,  -100,  -100,  -100,  -100],
#         [ -100,  -100, 14966,  -100,  -100,  -100]])

# indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool()

# print("indices_replaced:{}".format(indices_replaced))
# print("~indices_replaced:{}".format(~indices_replaced))

# masked_label = labels[labels != -100]

# print("labels:{}".format(labels))
# print("masked_label:{}".format(masked_label))
# print("masked_label.numel():{}".format(masked_label.numel()))

raw = '德城罪案调查处 主要任务 DBI的任务是调查最疑难最棘手的重案，支持法律'
jieba_appended = " ".join(jieba.lcut(raw.strip()))
print("jieba_appended: {}".format(jieba_appended))
new_appended = " ".join(raw.strip())
print("new_appended: {}".format(new_appended))


