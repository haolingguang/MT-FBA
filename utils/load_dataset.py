from torchvision import datasets,transforms
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, dirichlet_split_noniid
from utils.my_dataset import MyDataSet

def load_dataset(dataset_name, data_dir, iid = True, num_clients=None):

    if dataset_name == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(data_dir, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(data_dir, train=False, download=True, transform=trans_mnist)
        # sample users
        if iid:
            dict_users = mnist_iid(dataset_train, num_clients)
        else:
            dict_users = mnist_noniid(dataset_train, num_clients)
            
    elif dataset_name == 'cifar':
        trans_train = transforms.Compose(
            [transforms.ToTensor(), 
             transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
             transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 归一化到-1到1
        trans_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_test)
        if iid:
            dict_users = cifar_iid(dataset_train, num_clients)
        else:
            # dict_users = cifar_noniid(dataset_train, num_clients)
            dict_users = dirichlet_split_noniid(dataset_train, num_clients, 0.5)
    elif dataset_name == 'mini-imagenet':
        trans_train = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        trans_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        json_path = "classes_name.json"
        data_root = "../data/mini-imagenet/"
        dataset_train = MyDataSet(root_dir=data_root,
                              csv_name="new_train.csv",
                              json_path=json_path,
                              transform=trans_train)
        dataset_test = MyDataSet(root_dir=data_root,
                            csv_name="new_val.csv",
                            json_path=json_path,
                            transform=trans_test)
        if iid:
            dict_users = cifar_iid(dataset_train, num_clients)
        else:
            exit('Error: only consider IID setting in CIFAR10')
        
    else:
        exit('Error: unrecognized dataset')
    
    return dataset_train, dataset_test, dict_users


