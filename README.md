# MT-FBA


## Main Requirements
Ubuntu 20.04   
CUDA 11.8   
cudnn8   
python==2.10.13   
pytorch==2.0.1     
torchvision==0.15.2    
timm == 1.0.3    
opacus==1.4.1    
pandas==2.2.2    

## Run   
   
See the arguments in [options.py](utils/options.py).    
    
For example:   
> python main.py --dataset cifar --iid True --num_channels 3 --model 'Resnet18' --rounds 500  --gpu 0     
    
`--disable_dp` Differential privacy off    
'--disable_BA' Backdoor attack off     
   

   
## References   
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561


