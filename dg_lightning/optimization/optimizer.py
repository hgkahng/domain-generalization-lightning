
import typing

import torch
import torch.optim as optim


def create_optimizer(
    params: typing.Iterable,
    name: str,
    lr: float,
    weight_decay: typing.Optional[float] = 0.,
    **kwargs,
    ) -> optim.Optimizer:
    """
    Configure pytorch optimizer.
    Arguments:
        params: module parameters.
        name: str.
        lr: float.
        weight_decay: (optional) float.
        **kwargs: keyword arguments passed to the optimizer corresponding to `name`.
    Returns:
        a `torch.optim.Optimizer` instance.
    """
    
    name: str = name.lower()  # e.g., SGD -> sgd
    
    if name == 'sgd':
        return optim.SGD(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
        )
    
    elif name == 'adam':
        return optim.Adam(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            amsgrad=False,
        )
    
    elif name == 'adamw':
        return optim.AdamW(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
        )
    
    elif (name == 'lbfgs') or (name == 'l-bfgs'):
        return optim.LBFGS(
            params=params,
            lr=lr,
            line_search_fn='strong_wolfe',
            history_size=kwargs.get('history_size', 10),
            max_iter=kwargs.get('max_iter', 5),
        )
    
    elif name == 'lars':
        return LARS(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            eta=kwargs.get('eta', 0.001),
        )
    
    else:
        raise NotImplementedError(
            "Currently only supports one of [sgd, adam, adamw, lbfgs (or l-bfgs)]."
        )


class LARS(optim.Optimizer):
    def __init__(
        self,
        params: typing.Iterable,  # TODO: specify type hint
        lr: float = 0.2,
        weight_decay: float = 0.,
        momentum: float = 0.9,
        eta: float = 1e-3,
        ) -> None:
        
        # set defaults
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
        )

        super().__init__(params, defaults)

    @staticmethod
    def exclude_bias_and_norm(p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        
        for g in self.param_groups:
            
            for p in g['params']:
                
                dp = p.grad
                if dp is None:
                    continue

                if not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])
                
                if not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    _ones = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0.0,
                            g['eta'] * param_norm / update_norm,
                            _ones
                        ),
                        _ones
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
