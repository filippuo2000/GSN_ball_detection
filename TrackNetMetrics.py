from torchmetrics import Metric
import torch

# it does not detect false positives yet
class MyMetrics(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("precision", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("recall", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("accuracy", default=torch.tensor(0), dist_reduce_fx="sum")

    # expects tensors of shape [Batch_size, 2] 2 because x and y is needed
    #
    def update(self, preds, target) -> None:
        # CHECK if this is correct !!!!
        self.total = target[0].shape[0]  # batch size

        target = torch.stack(target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")


        # set [-10,-10] for not detected in preds
        # set [-100, -100] for not detected in target
        self.incorrect = (torch.sqrt(((preds-target)**2).sum(dim=0))>5).sum() # helper
        self.tp = (torch.sqrt(((preds-target)**2).sum(dim=0))<=5).sum() # TP

        preds_ = preds.sum(dim=0)
        target_ = target.sum(dim=0)

        mask_fn = (preds_ < 0) & (target_ > 0) # not detected when should have
        mask_fp = (preds_ > 0) & (target_ < 0) # detected when it should not have
        mask_tn = (preds_ < 0) & (target_ < 0) # not detected when should not have
        self.fn = mask_fn.sum() # FN
        self.tn = mask_tn.sum() # TN
        self.fp = self.incorrect - self.tn - self.fn # FP


    def compute(self) -> torch.Tensor:
        return {
            "precision": self.tp.float() / (self.tp.float() + self.fp.float() + 0.00001),
            "recall": self.tp.float() / (self.tp.float() + self.fn.float() + 0.000001),
            "accuracy": (self.tp.float() + self.tn.float()) / self.total
        }

# precision = TP / TP+FP
# recall = TP / TP+FN
# TP - when ball is detected correctly
# FP - when there's no ball, but it is detected or the ball is detected incorrectly
# FN - when the ball is present, but not detected