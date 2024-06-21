from models.Nets import MLP, CNNMnist, CNNCifar
import timm
import torch
def load_model(model_name, dataset_name, img_size,num_classes):
    # build model
    if dataset_name == 'cifar':
        net_glob = timm.create_model(model_name, pretrained=False, num_classes=10 )
        net_glob.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False) 
        net_glob.maxpool = torch.nn.MaxPool2d(1, 1, 0)  
        net_glob = net_glob.cuda()
    elif dataset_name == 'mnist':
        net_glob = CNNMnist(num_classes).cuda()
    elif model_name == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=num_classes).cuda()
    else:
        exit('Error: unrecognized model')
    return net_glob