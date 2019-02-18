import torch
# Convert annotation mask to one hot encoding
def LabelConvert(label,NumClasses=2):
    label = torch.from_numpy(label).cuda()
    # create one-hot encoding
    batchsize, h, w= label.size()
    target = torch.zeros(batchsize,NumClasses, h, w).cuda()
    for c in range(NumClasses):
       for b in range(batchsize):
         target[b][c][label[b] == c] = 1
    return torch.autograd.Variable(target,requires_grad=False)
