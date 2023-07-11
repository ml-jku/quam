import torch
from torch import nn


def _raise_exception(*args, **kwargs) -> None:
    raise Exception("Can't backward")


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.register_full_backward_hook(_raise_exception)

    def forward(self, y_hat, y):
        '''

        :param y_hat:
        [B, *, O]
        :param y:
        [B, *]
        :return:
        '''
        _, y_hat_pred = torch.max(y_hat, dim=-1)
        return torch.sum(y_hat_pred == y, dim=0)/y.shape[0]

