from torchmetrics import Metric
import torch

# it does not detect false positives yet
class MyMetrics(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    # expects tensors of shape [Batch_size, 2] 2 because x and y is needed
    def update(self, preds, target) -> None:
        # CHECK if this is correct !!!!
        self.total += target[0].shape[0]  # batch size

        target = torch.stack(target)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        #print("false negatives: ", mask_fn.sum())
        #print("false positive: ", mask_fp.sum())
        #print("true negative: ", mask_tn.sum())

        # set [-10,-10] for not detected in preds
        # set [-100, -100] for not detected in target
        incorrect = (torch.sqrt(((preds-target)**2).sum(dim=0))>10).sum() # helper
        self.true_positives += (torch.sqrt(((preds-target)**2).sum(dim=0))<=10).sum() # TP

        preds_ = preds.sum(dim=0)
        target_ = target.sum(dim=0)

        mask_fn = (preds_ < 0) & (target_ > 0) # not detected when should have
        mask_fp = (preds_ > 0) & (target_ < 0) # detected when it should not have
        mask_tn = (preds_ < 0) & (target_ < 0) # not detected when should not have
        self.false_negatives += mask_fn.sum() # FN
        self.true_negatives += mask_tn.sum() # TN
        self.false_positives += incorrect - mask_tn.sum() - mask_fn.sum() # FP


    def compute(self):
        #print('total:\t', self.total)
        #print('true_positives:\t', self.true_positives)
        #print('true_negatives:\t', self.true_negatives)
        #print('false_positives:\t', self.false_positives)
        #print('false_negative:\t', self.false_negatives)
        accuracy = (self.true_positives + self.true_negatives) / self.total
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-10)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-10)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }

    def reset(self):
        # Reset the state variables
        self.true_positives = torch.tensor(0.0)
        self.false_positives = torch.tensor(0.0)
        self.false_negatives = torch.tensor(0.0)
        self.true_negatives = torch.tensor(0.0)
        self.total = torch.tensor(0.0)

# precision = TP / TP+FP
# recall = TP / TP+FN
# TP - when ball is detected correctly
# FP - when there's no ball, but it is detected or the ball is detected incorrectly
# FN - when the ball is present, but not detected