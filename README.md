# MT-FBA: Multi-Target Federated Backdoor Attack Based on Feature Aggregation

## Abstract   
Current federated backdoor attacks focus on collaboratively training backdoor triggers, where multiple compromised clients train their local trigger patches and then merge them into a global trigger during the inference phase. However, these methods require careful design of the shape and position of trigger patches and lack the feature interactions between trigger patches during training, resulting in poor backdoor attack success rates. Moreover, the pixels of the patches remain untruncated, thereby making abrupt areas in backdoor examples easily detectable by the detection algorithm. To this end, we propose a novel benchmark for the federated backdoor attack based on feature aggregation. Specifically, we align the dimensions of triggers with images, delimit the trigger’s pixel boundaries, and facilitate feature interaction among local triggers trained by each compromised client. Furthermore, leveraging the intraclass attack strategy, we propose the simultaneous generation of backdoor triggers for all target classes, significantly reducing
the overall production time for triggers across all target classes and increasing the risk of the federated model being attacked. Experiments demonstrate that our method can not only bypass the detection of defense methods while patch-based methods fail, but also achieve a zero-shot backdoor attack with a success rate of 77.39%. To the best of our knowledge, our work is the first to implement such a zero-shot attack in federated learning. Finally, we evaluate attack performance by varying the trigger’s training factors, including poison location, ratio, pixel bound, and trigger training duration (local epochs and communication rounds).   

## Main Environment  
> Ubuntu 20.04  
> CUDA 11.8  
> cudnn8  
> python==3.10.13  
> pytorch==2.0.1  
> torchvision==0.15.2  
> timm == 1.0.3   
> opacus==1.4.1  
> pandas==2.2.2      

## Download mini-ImageNet 
[Baidu_cloud](https://pan.baidu.com/s/1KKk7O418DoVFeFrs4tLt-A?pwd=uf0z)    
[Google_drive](https://drive.google.com/file/d/1qnytH4OCjuK9kOq5-32c9RfSdOoJI_gj/view?usp=drive_link)    

## Run   
   
See the arguments in [options.py](utils/options.py).    
    
For example:   
> python main.py --dataset 'cifar' --iid True --num_channels 3 --model 'resnet18' --rounds 500  --gpu '0'          
> python test_backdoor_image.py --defense 'NRP' --dataset 'cifar' --num_classes 10 --model 'resnet18'  --gpu '0'           
    
`--disable_dp` Differential privacy off    
`--disable_BA`   Backdoor attack off     
   
if use Differential privacy, set options:  --disable_dp False  --local_bs 500  --lr 0.01        
> python main.py --disable_dp False  --local_bs 500   --lr 0.01  --dataset 'cifar' --iid True --num_channels 3 --model 'resnet18' --rounds 500  --gpu '0'      

## References   
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561     


