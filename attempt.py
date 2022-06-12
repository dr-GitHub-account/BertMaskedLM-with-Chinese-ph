import torch

labels = torch.tensor([[ -100,  -100,  -100,  -100,  -100,  -100],
        [ -100,  -100,  -100,  -100,  -100,  -100],
        [ -100,  -100,  -100,  -100,  -100,  -100],
        [ -100,  -100,  -100,  -100,  -100,  -100],
        [ -100,   704,  -100,  -100,  -100,  -100],
        [ -100,  -100, 14966,  -100,  -100,  -100]])

masked_label = labels[labels != -100]

print("labels:{}".format(labels))
print("masked_label:{}".format(masked_label))
print("masked_label.numel():{}".format(masked_label.numel()))

