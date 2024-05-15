from torchmetrics import Metric
#from callable import List
import torch

# it does not detect false positives yet
class MyAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    # expects tensors of shape [Batch_size, 2] 2 because x and y is needed
    #
    def update(self, preds, target) -> None:
        target = torch.stack(target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.correct = (torch.sqrt((preds-target)**2).sum(dim=0)<=5).sum()

        # CHECK if this is correct !!!!
        self.total += target.shape[1] # batch size, but

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total