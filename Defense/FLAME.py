import torch
import hdbscan



def flame_defense(local_models, global_model):

# 我们的这个本地模型参数就是训练结束后的参数，不是差值，和参考的代码有所不同！！！！

    # 用来存储筛选后模型参数聚合
    weight_accumulator = {}
    for name, params in global_model.items():
        weight_accumulator[name] = torch.zeros_like(params)

    ############ 0. 数据预处理，将clients_weight展开成二维tensor, 方便聚类计算
    clients_weight_ = []
    clients_weight_total = []
    for data in local_models:
        client_weight = torch.tensor([])
        client_weight_total = torch.tensor([])

        for name, params in data.items():
            client_weight = torch.cat((client_weight, params.reshape(-1).cpu()))
            if name == 'fc.weight' or name == 'fc.bias':
                client_weight_total = torch.cat((client_weight_total, params.reshape(-1).cpu()))
                # 注意上面这行代码进行了改动，因为我们的本地模型参数直接是训练后的模型
                
        clients_weight_.append(client_weight)
        clients_weight_total.append(client_weight_total)
        
    # 对全局模型也展开方便裁剪时计算欧氏距离
    global_weight = torch.tensor([])
    for name, params in global_model.items():
        global_weight = torch.cat((global_weight, params.reshape(-1).cpu()))
    
    ############ 1. 利用HDBSCAN排除异常值
    clients_weight_total = torch.stack(clients_weight_total)
    clients_weight_total = clients_weight_total.double()  # 这里复现用的全连接层？
    cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic", min_cluster_size=len(local_models)//2+1, min_samples=1,allow_single_cluster=True)
    cluster.fit(clients_weight_total)

    ############2. 范数中值裁剪
    diff = torch.stack([(loc_weight-global_weight) for loc_weight in clients_weight_])
    euclidean = (diff ** 2).sum(1).sqrt()
    med = euclidean.median()

    # 对本地模型的参数进行裁剪，只计算聚类保留下来的模型参数，异常值不计算节省算力
    for i, data in enumerate(local_models):
        if cluster.labels_[i] == 0:
            gama = med.div(euclidean[i])
            if gama > 1:
                gama = 1
            for name, params in data.items():
                local_models[i][name]=((params.data-global_model[name]) * gama \
                               + global_model[name])
                    

    ############3. 聚合
    # 累加所有的聚类后剩下来的模型参数
    num_in = 0
    for i, data in enumerate(local_models):		
        if cluster.labels_[i] == 0:
            num_in += 1
            for name, params in data.items():
                weight_accumulator[name].add_(params.to(weight_accumulator[name].data.dtype))
    # 更新全局模型
    # t = global_model.state_dict().items()
    for name, params in global_model.items():
        global_model[name] = weight_accumulator[name] / num_in
        # print(global_model[name])

    ############4. 添加噪声
    lamda = 0.000012
    for name, param in global_model.items():
        if 'bias' in name or 'bn' in name:
			# 不对偏置和BatchNorm的参数添加噪声
            continue
        std = lamda * med 
        noise = torch.normal(0, std, size=param.size()).cuda()
        param.data.add_(noise)
    return global_model
    
   