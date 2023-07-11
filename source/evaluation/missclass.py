import torch
from .ood import evaluate_score


def evaluate_missclass(
    id_uncerts,
    id_uncerts_target,
    id_uncerts_average_net_pred,
):
    # compute the "accuracy score" - 1 if correct, 0 if incorrect
    #print(id_uncerts_average_net_pred, id_uncerts_target)
    accuracy_scores = torch.argmax(id_uncerts_average_net_pred, dim=-1) != id_uncerts_target # [n_points]
    # the correct predictions - class 0, should be associated with lower uncertainty values\
    # the incorrect - class 1 with higher uncertainties
    return evaluate_score(accuracy_scores, id_uncerts.float())
