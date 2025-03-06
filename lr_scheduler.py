from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer


class CustomScheduler(LRScheduler):
    '''Learning rate scheduler as defined in "Attention Is All You Need". '''

    def __init__(
        self, 
        optimizer: Optimizer, 
        d_model: int, 
        warmup_steps: int = 4000
    ):
        
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        super().__init__(optimizer)


    def get_lr(self) -> list[float]:

        lr = self.d_model**-0.5 * min(
            (self._step_count + 1)**-0.5, 
            (self._step_count + 1) * self.warmup_steps**-1.5
        )

        return [lr] * len(self.optimizer.param_groups)