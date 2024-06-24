import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import os
from torchvision.utils import save_image

from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import test_img,test_backdoor
from models.train_trigger import generate_trigger


def Fed_train(args, net_glob, dataset_train, dict_users, dataset_test, log_string):
    
    # copy global model parameters
    w_glob = net_glob.state_dict()
    net_glob.eval()
    
    # loss and accuracy 
    loss_train = []
    acc_train = []
    acc_test = []
    loss_glob = 10.0

    # generate poison client's numbers   
    poison_clients = set(np.random.choice(args.num_clients, args.num_poison, replace=False))
    
    trigger = None
    # train
    for iter in range(args.rounds):  
        loss_locals = []
        
        # local client's model parameters list
        w_locals = []
        
        # Using a near-convergent model to poison
        if iter == args.point and not args.disable_BA:
            trigger = generate_trigger(args, dataset_train, dict_users, poison_clients,net_glob)
            Save_trigger(copy.deepcopy(trigger), './save/Trigger_image')

        # server select client
        m = max(int(args.frac * args.num_clients), 1)   
        idxs_users = np.random.choice(range(args.num_clients), m, replace=False)
      
        # walk through each client and train
        for idx in idxs_users:
            
            # Defines client objects
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            # 
            if (idx in poison_clients) and (iter>args.point):
                w, loss = local.train_backoodr_model(copy.deepcopy(net_glob).cuda(), net_glob, copy.deepcopy(trigger), rounds=iter)        
            else:  
                w, loss = local.train(net=copy.deepcopy(net_glob).cuda(), rounds=iter)
            
            # logging client model parameters
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            
        # aggregate
        w_glob = FedAvg(w_locals)

        # update global model parameters
        if args.disable_dp:
            net_glob.load_state_dict(w_glob)
        else:
            net_glob.load_state_dict({k.replace('_module.',''):v for k,v in w_glob.items()})
        # 
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        # save model weight
        if loss_glob>loss_avg:
            loss_glob=loss_avg
            torch.save(net_glob.state_dict(), './checkpoints/{}.pth'.format(args.model))
            
        # compute accuracy
        train_glob, _ = test_img(net_glob, dataset_train, args)
        test_glob, _ = test_img(net_glob, dataset_test, args)
        acc_train.append(train_glob)        
        acc_test.append(test_glob)
        
    # logging
    # max(acc_train)
    # max(acc_test)
    log_string("Training accuracy: {:.2f}".format(train_glob))
    log_string("Testing accuracy: {:.2f}".format(test_glob))
    
    # test backdoor
    if not args.disable_BA:
        for i in range(10):
            attack_succ_rate, _ =test_backdoor(net_glob, dataset_test, args, trigger, i) 
            log_string("Label_{} Attack accuracy: {:.2f}".format(i,attack_succ_rate))

    # plot training loss
    plot_acc(acc_train, acc_test, './save/acc_{}_{}_{}_C{}_iid{}.pdf'.format(args.dataset,args.model,args.rounds,args.frac,args.iid))    
    plot_loss(loss_train,'./save/loss_{}_{}_{}_C{}_iid{}.pdf'.format(args.dataset,args.model,args.rounds,args.frac,args.iid))
    return net_glob


def plot_loss(data, path):

    # 
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
    
    # 
    xAxis1 = range(0,len(data))
    plt.plot(xAxis1, data, color='blue', linestyle='-',linewidth=1)
    
    # 
    plt.xlabel('Rounds',fontsize=15)
    plt.ylabel('Training Loss',fontsize=15)
    
    # 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # 
    ax=plt.gca()
    
    # 
    plt.savefig(path,dpi=500,bbox_inches = 'tight')
    plt.show()
    plt.clf()


def plot_acc(train_acc, test_acc, path):
    
    #
    plt.rcParams['font.sans-serif'] = ['Times New Roman']   # font
    plt.rcParams['axes.unicode_minus'] = False   # -
    plt.rcParams['xtick.direction'] = 'in' # 
    plt.rcParams['ytick.direction'] = 'in' #

    # plot curve
    xAxis1 = range(0,len(train_acc))
    plt.plot(xAxis1, train_acc, color='blue', linestyle='-',linewidth=1, label='Train' )
    plt.plot(xAxis1, test_acc, color='red', linestyle='-',linewidth=1, label='Test')

    # Sets the name of the axis
    plt.xlabel('Rounds',fontsize=15)
    plt.ylabel('Accuracy',fontsize=15)
    
    # set the title of figure
    # plt.title("InceptionResnet-v2",fontsize=15)
    
    # set size of font 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    # Set the axis scale interval
    ax=plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(200))
    # ax.yaxis.set_major_locator(MultipleLocator(y_locator))
    # # plt.xlim(0,11)
    # plt.ylim(-y_clim,y_clim)
    
    # Set the position and font size of the legend
    plt.legend(loc='lower right', fontsize=15)
    
    #set grid
    # plt.grid(axis="y", linewidth=0.1)
    
    # save figure
    plt.savefig(path,dpi=500, bbox_inches='tight')
    plt.show()
    plt.clf()


def Save_trigger(trigger, output_dir):
    # 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    for index in range(len(trigger)):

        trigger_norm = trigger[index].div_(2).add(0.5)
        trigger_path = os.path.join(output_dir, 'Trigger'+str(index)+'.png')
        save_image(trigger_norm, trigger_path)