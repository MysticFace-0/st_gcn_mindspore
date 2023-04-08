import mindspore
import numpy as np
import torch

def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res

x = torch.Tensor([[1,2,3,3,4,2],[1,2,3,4,4,5],[6,5,4,3,2,1],[1,2,3,4,5,6]]).numpy()
y = torch.Tensor([4,5,6,1]).numpy()

res = top_k_accuracy(x,y,(1,))
print(res)



