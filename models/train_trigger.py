import torch
from models.Update import LocalUpdate
import copy


def generate_trigger(args, dataset_train, dict_users, poison_clients,net_glob):
   
    # Define a list of trigger sizes of length 10, where each element is a tensor of 1*3*32*32
    noise_glob = [torch.zeros(args.image_size[0], args.image_size[1], args.image_size[2]).cuda() for i in range(args.num_classes)]
       
    # Here the trigger training  uses a federated approach
    for e in range(args.trigger_round):
        noise_locals = []

        # deepcopy
        for idx in poison_clients: 
            l_noise_glob = copy.deepcopy(noise_glob)
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            noise = local.train_noise(net_glob=copy.deepcopy(net_glob).cuda(), noise=l_noise_glob)
            noise_locals.append(copy.deepcopy(noise))

        # trigger aggregate
        noise_glob_data = Fed_trigger(noise_locals)
        for i in range(len(noise_glob)):
            noise_glob[i].data = noise_glob_data[i].data
        
    return noise_glob

def Fed_trigger(noise_local):
    ''' noise_local is a list with a length of the number of clients, and each element is a list of num_classes.
    Each element in the list is the noise tensor of [3,32,32] '''
    
    noise_avg = copy.deepcopy(noise_local[0])
    
    for k in range(len(noise_avg)):
        for i in range(1,len(noise_local)):
            noise_avg[k] = noise_avg[k] + noise_local[i][k]
        noise_avg[k] = torch.div(noise_avg[k], len(noise_local))
    return noise_avg

