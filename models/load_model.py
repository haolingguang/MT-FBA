from models.Nets import resnet18
import timm
import torch
def load_model(model_name, dataset_name, num_classes, disable_dp=True):
    # build model
    if dataset_name == 'cifar':
        if not disable_dp:
            net_glob = resnet18()
        else:
            net_glob = timm.create_model(model_name, pretrained=False, num_classes=10 )
            net_glob.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False) 
            net_glob.maxpool = torch.nn.MaxPool2d(1, 1, 0)  
        net_glob = net_glob.cuda()
    return net_glob