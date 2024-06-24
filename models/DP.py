from opacus import PrivacyEngine


# Difference pravicy setting
class Differencial_Privacy(object):
    '''
        max_per_sample_grad_norm: Clip per-sample gradients to this norm (default 1.0)
        secure-rng: Enable Secure RNG to have trustworthy privacy guarantees. \
                Comes at a performance cost. Opacus will emit a warning if secure rng is off, \
                indicating that for production use it's recommender to turn it on.
        sigma: Noise multiplier (default 1.0)
        delta: Target delta (default: 1e-5)
        clip_per_layer: Clip per-sample gradients to this norm (default 1.0)
        grad_sample_mode: obtain grad by mode "hook"
        dataloader: train data loader
        optimizer: model optimizer
    '''

    def __init__(self, max_per_sample_grad_norm=10.0, secure_rng=False, sigma=1.5, delta=1e-5, 
                 clip_per_layer = False, grad_sample_mode="hooks"):
        
        self.max_grad_norm = max_per_sample_grad_norm
        self.clipping = "per_layer" if clip_per_layer else "flat"
        self.privacy_engine = PrivacyEngine(secure_mode=secure_rng)
        self.grad_sample_mode = grad_sample_mode
        self.sigma = sigma
        self.delta = delta


    def init_model(self, dataloader=None, optimizer=None, net=None):


        net, optimizer, dataloader = self.privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=self.sigma,
            max_grad_norm=self.max_grad_norm,
            clipping=self.clipping,
            grad_sample_mode=self.grad_sample_mode
        )
        return net, optimizer, dataloader

    def DP_epsilon(self):
        return self.privacy_engine.accountant.get_epsilon(delta=self.delta)