import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img



def Fed_train(args, net_glob, dataset_train, dict_users, dataset_test, log_string):
    
    # copy weights
    w_glob = net_glob.state_dict()
    net_glob.eval()
    
    # training 
    loss_train = []
    acc_train = []
    acc_test = []
    loss_glob = 1.0
        
    # 遍历通信轮数进行训练，这里的epochs是rounds，即客户端与服务器通信轮数
    for iter in range(args.rounds):  
        loss_locals = []
        
        w_locals = []
        fracs = []    
        # 一次选择100个客户端中的10%个
        m = max(int(args.frac * args.num_clients), 1)   
        idxs_users = np.random.choice(range(args.num_clients), m, replace=False)
        
        # 遍历每个客户端并进行训练
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).cuda())
            
            fracs.append(len(dict_users[idx]))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
        # 将所有客户端模型参数进行聚合
        w_glob = FedAvg(w_locals,fracs)

        # 更新全局模型参数
        net_glob.load_state_dict(w_glob)

        # 打印每个round训练的平均loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        # 保存模型参数
        if loss_glob>loss_avg:
            loss_glob=loss_avg
            torch.save(net_glob.state_dict(), './checkpoints/{}/{}.pth'.format(args.dataset,args.model))
        # 计算全局模型在训练集和测试集的精度
        net_glob.eval()
        train_glob, _ = test_img(net_glob, dataset_train, args)
        test_glob, _ = test_img(net_glob, dataset_test, args)
        acc_train.append(train_glob)        
        acc_test.append(test_glob)
        
    # 记录最后一次精度的日志
    log_string("{}_Training accuracy: {:.2f}".format(args.model, train_glob))
    log_string("{}_Testing accuracy: {:.2f}".format(args.model, test_glob))
    

    # 绘制总的精度曲线和loss曲线
    plot_acc(acc_train, acc_test, './save/{}/acc_{}_{}_{}_C{}_iid{}.pdf'.format(args.dataset,args.dataset,args.model,args.rounds,args.frac,args.iid))    
    plot_loss(loss_train,'./save/{}/loss_{}_{}_{}_C{}_iid{}.pdf'.format(args.dataset,args.dataset,args.model,args.rounds,args.frac,args.iid))
    return net_glob

def plot_loss(data, path):

    #调节字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
    # 画数据
    t = data
    xAxis1 = range(0,len(t))
    plt.plot(xAxis1, t, color='blue', linestyle='-',linewidth=1)
    # 定义坐标轴的名称
    plt.xlabel('Rounds',fontsize=15)
    plt.ylabel('Training Loss',fontsize=15)
    # 定义坐标轴字体大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # 画网格
    ax=plt.gca()
    plt.savefig(path,dpi=500,bbox_inches = 'tight')
    plt.show()
    plt.clf()

def plot_acc(train_acc, test_acc, path):
    
    #调节字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman']   # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号
    plt.rcParams['xtick.direction'] = 'in' #将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in' #将y轴的刻度方向设置向内

    xAxis1 = range(0,len(train_acc))
    plt.plot(xAxis1, train_acc, color='blue', linestyle='-',linewidth=1, label='Train' )
    plt.plot(xAxis1, test_acc, color='red', linestyle='-',linewidth=1, label='Test')

    plt.xlabel('Rounds',fontsize=15)
    plt.ylabel('Accuracy',fontsize=15)
    # plt.title("InceptionResnet-v2",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #设置坐标轴刻度间隔
    ax=plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(200))
    # ax.yaxis.set_major_locator(MultipleLocator(y_locator))
    # # plt.xlim(0,11)
    # plt.ylim(-y_clim,y_clim)
    # 画图例
    plt.legend(loc='lower right', fontsize=15)
    #设置网格
    # plt.grid(axis="y", linewidth=0.1)
    plt.savefig(path,dpi=500, bbox_inches='tight')
    plt.show()
    plt.clf()
