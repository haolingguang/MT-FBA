from models.Nets import MLP, CNNMnist, CNNCifar
import timm
import torch
def load_model(model_name, dataset_name, img_size,num_classes):
    # build model
    if dataset_name == 'cifar':
        # print(timm.list_models(pretrained=True))
        net_glob = timm.create_model(model_name, pretrained=False, num_classes=num_classes )
        net_glob.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
        net_glob.maxpool = torch.nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
         # num_ftrs = net_glob.fc.in_features  # 获取（fc）层的输入的特征数
        # net_glob.fc = torch.nn.Linear(num_ftrs, 10)
        if model_name == 'inception_v3':
            net_glob.Pool1 = torch.nn.MaxPool2d(1, 1, 0)
            net_glob.Pool2 = torch.nn.MaxPool2d(1, 1, 0)
        if model_name == 'inception_resnet_v2':
            net_glob.maxpool_3a = torch.nn.MaxPool2d(1, 1, 0)
            net_glob.maxpool_5a = torch.nn.MaxPool2d(1, 1, 0)
        net_glob = net_glob.cuda()

    elif dataset_name == 'mnist':
        net_glob = timm.create_model(model_name, pretrained=False, num_classes=num_classes )
        net_glob.conv1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)  # 首层改成3x3卷积核
        net_glob.maxpool = torch.nn.MaxPool2d(1, 1, 0)  # 图像太小 本来就没什么特征 所以这里通过1x1的池化核让池化层失效
        net_glob = net_glob.cuda()
        
    elif model_name == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=num_classes).cuda()
    else:
        exit('Error: unrecognized model')
    return net_glob