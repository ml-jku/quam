import torch
from torch import nn

from typing import Callable, List, Tuple

from copy import deepcopy

from abc import ABC, abstractmethod, abstractproperty


def _raise_exception(*args, **kwargs) -> None:
    raise Exception("Can't backward")


class NDModule(ABC):
    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def check_compat(self, m: nn.Module) -> bool:
        pass

    @abstractmethod
    def mirror_nth_member(self, idx: int, other):
        pass

    @abstractmethod
    def get_nth_member(self, idx: int) -> nn.Module:
        pass

    @abstractmethod
    def set_nth_member(self, idx: int, member: nn.Module):
        pass

    def unbatch(self) -> List[nn.Module]:
        return [self.get_nth_member_as_linear(i) for i in range(self.n)]


class LinearND(nn.Module, NDModule):
    def __init__(self, linear: nn.Linear, n: int) -> None:
        super(LinearND, self).__init__()

        device = linear.weight.device
        dtype = linear.weight.dtype

        self.multiplicity = n

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight = nn.Parameter(linear.weight.data.unsqueeze(0).repeat(n, 1, 1).to(device).to(dtype))
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.unsqueeze(0).repeat(n, 1).to(device).to(dtype))
        return

    @property
    def n(self):
        return self.multiplicity

    def check_compat(self, m: nn.Module) -> bool:
        if isinstance(m, nn.Linear):
            return True
        else:
            return False

    @torch.no_grad()
    def get_nth_member(self, idx: int) -> nn.Linear:
        retval = nn.Linear(self.in_features, self.out_features, bias=hasattr(self, 'bias'))

        retval.weight.data = self.weight.data[idx, :, :]
        if hasattr(self, 'bias'):
            retval.bias.data = self.bias.data[idx, :]

        return retval

    @torch.no_grad()
    def set_nth_member(self, idx: int, member: nn.Linear):
        self.weight.data[idx, :, :] = member.weight.data
        if hasattr(self, 'bias'):
            self.bias.data[idx, :] = member.bias.data

    @torch.no_grad()
    def mirror_nth_member(self, idx: int, other):
        self.weight.data[idx, :, :] = other.weight.data[idx, :, :]
        if hasattr(self, 'bias'):
            self.bias.data[idx, :] = other.bias.data[idx, :]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input is either [n, b, in] or [b, in*n] or invalid
        if input.ndim == 2:
            input = input.reshape(input.shape[0], self.n, -1).transpose(0, 1)
        # [n, out, in] x [n, b, in]-> [n, b, out]
        # print(input.shape, self.weight.data.shape, self.bias.shape)
        return torch.matmul(input, self.weight.transpose(1, 2)).add(self.bias.unsqueeze(1)) if hasattr(self,
                                                                                                       'bias') else torch.matmul(
            input, self.weight.transpose(1, 2))


class Conv2DND(nn.Conv2d, NDModule):
    def __init__(self, conv: nn.Conv2d, n_groups: int):
        # preconfigure
        device = next(conv.parameters()).device
        dtype = next(conv.parameters()).dtype
        has_bias = hasattr(conv, 'bias') and conv.bias is not None

        self.multiplicity = n_groups

        # initialize own params
        super().__init__(
            conv.in_channels * n_groups,
            conv.out_channels * n_groups,
            conv.kernel_size,
            bias=has_bias,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            device=device,
            dtype=dtype,
            groups=n_groups,
        )

        # the weights are to be concatenated on the output dimension
        nuweights = conv.weight.data.repeat(n_groups, 1, 1, 1)

        assert nuweights.shape == self.weight.data.shape, 'Something is up with repeating the weights!'

        self.weight.data = nuweights

        if has_bias:
            nubias = conv.bias.data.repeat(n_groups)
            assert nubias.shape == self.bias.data.shape, 'Something is up with repeating the weights the biases!'
            self.bias.data = nubias

    @property
    def n(self):
        return self.multiplicity

    def check_compat(self, m: nn.Module) -> bool:
        if isinstance(m, nn.Conv2d):
            return True
        else:
            return False

    @torch.no_grad()
    def mirror_nth_member(self, idx: int, other):
        n_in = self.in_channels // self.n
        n_out = self.out_channels // self.n
        self.weight.data[n_out * idx:n_out * (idx + 1), :, :, :] = other.weight.data[n_out * idx:n_out * (idx + 1), :,
                                                                   :, :]
        if hasattr(self, 'bias') and self.bias is not None:
            self.bias.data[n_out * idx: n_out * (idx + 1)] = other.bias.data[n_out * idx: n_out * (idx + 1)]

    @torch.no_grad()
    def get_nth_member(self, idx: int) -> nn.Module:
        n_in = self.in_channels // self.n
        n_out = self.out_channels // self.n

        retval = nn.Conv2d(
            self.in_channels // self.n,
            self.out_channels // self.n,
            self.kernel_size,
            bias=hasattr(self, 'bias'),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            device=self.weight.data.device,
            dtype=self.weight.data.dtype,
            # groups=1,
        )

        retval.weight.data = self.weight.data[n_out * idx: n_out * (idx + 1)]

        if hasattr(self, 'bias') and self.bias is not None:
            retval.bias.data = self.bias.data[n_out * idx: n_out * (idx + 1)]

        return retval

    @torch.no_grad()
    def set_nth_member(self, idx: int, member: nn.Conv2d):
        n_in = self.in_channels // self.n
        n_out = self.out_channels // self.n

        self.weight.data[n_out * idx: n_out * (idx + 1)] = member.weight.data

        if hasattr(self, 'bias') and self.bias is not None:
            self.bias.data[n_out * idx: n_out * (idx + 1)] = member.bias.data


class BN2DND(nn.BatchNorm2d, NDModule):
    def __init__(self, bn2d: nn.BatchNorm2d, n_groups: int):
        device = next(bn2d.parameters()).device
        dtype = next(bn2d.parameters()).dtype

        self.multiplicity = n_groups

        super().__init__(
            bn2d.num_features * n_groups,
            eps=bn2d.eps,
            momentum=bn2d.momentum,
            affine=bn2d.affine,
            track_running_stats=bn2d.track_running_stats,
            device=device,
            dtype=dtype,
        )

        nuweights = bn2d.weight.data.repeat(n_groups)
        assert nuweights.shape == self.weight.data.shape, 'Something is up with repeating the weights!'
        self.weight.data = nuweights

        nubias = bn2d.bias.data.repeat(n_groups)
        assert nubias.shape == self.bias.data.shape, 'Something is up with repeating the biases!'
        self.bias.data = nubias

        if bn2d.track_running_stats:
            self.running_mean.data = bn2d.running_mean.data.repeat(n_groups)
            self.running_var.data = bn2d.running_var.data.repeat(n_groups)

    @property
    def n(self):
        return self.multiplicity

    @torch.no_grad()
    def check_compat(self, m: nn.Module) -> bool:
        if isinstance(m, nn.BatchNorm2d):
            return True
        else:
            return False

    @torch.no_grad()
    def mirror_nth_member(self, idx: int, other):
        n_in = self.num_features // self.n

        self.weight.data[n_in * idx: n_in * (idx + 1)] = other.weight.data[n_in * idx: n_in * (idx + 1)]
        self.bias.data[n_in * idx: n_in * (idx + 1)] = other.bias.data[n_in * idx: n_in * (idx + 1)]

        if self.track_running_stats:
            self.running_mean.data[n_in * idx: n_in * (idx + 1)] = other.running_mean.data[n_in * idx: n_in * (idx + 1)]
            self.running_var.data[n_in * idx: n_in * (idx + 1)] = other.running_var.data[n_in * idx: n_in * (idx + 1)]

    @torch.no_grad()
    def get_nth_member(self, idx: int) -> nn.Module:
        n_in = self.num_features // self.n

        retval = nn.BatchNorm2d(
            n_in,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            device=self.weight.data.device,
            dtype=self.weight.data.dtype,
        )
        retval.weight.data = self.weight.data[n_in * idx: n_in * (idx + 1)]
        retval.bias.data = self.bias.data[n_in * idx: n_in * (idx + 1)]

        if self.track_running_stats:
            retval.running_mean.data = self.running_mean.data[n_in * idx: n_in * (idx + 1)]
            retval.running_var.data = self.running_var.data[n_in * idx: n_in * (idx + 1)]

        return retval

    @torch.no_grad()
    def set_nth_member(self, idx: int, member: nn.Module):

        self.weight.data[n_in * idx: n_in * (idx + 1)] = member.weight.data
        self.bias.data[n_in * idx: n_in * (idx + 1)] = member.bias.data

        if self.track_running_stats:
            n_in = self.num_features // self.n

            self.running_mean.data[n_in * idx: n_in * (idx + 1)] = member.running_mean.data
            self.running_var.data[n_in * idx: n_in * (idx + 1)] = member.running_var.data


def set_submodule(net: nn.Module, submod_path, replace_with: nn.Module):
    submod_list = submod_path.rsplit('.')

    def recursive_set(m: nn.Module, track: List[str], nmod: nn.Module):
        sub = track.pop(0)
        if len(track) == 0:
            if isinstance(m, nn.Sequential):
                m[int(sub)] = nmod
            else:
                m.__setattr__(sub, nmod)
            return
        if isinstance(m, nn.Sequential):
            sm = m[int(sub)]
        else:
            sm = m.__getattr__(sub)
        return recursive_set(sm, track, nmod)

    return recursive_set(net, submod_list, replace_with)


@torch.no_grad()
def batch_that_net(net: nn.Module, batch_size) -> nn.Module:
    batchmod = deepcopy(net)
    for n, m in net.named_modules():
        if isinstance(m, nn.Linear):
            set_submodule(batchmod, n, LinearND(m, batch_size))
        elif isinstance(m, nn.Conv2d):
            set_submodule(batchmod, n, Conv2DND(m, batch_size))
        elif isinstance(m, nn.BatchNorm2d):
            set_submodule(batchmod, n, BN2DND(m, batch_size))
        else:
            continue

    return batchmod


@torch.no_grad()
def mirror_member_net(dest: nn.Module, idx: int, src: nn.Module):
    for n, m in dest.named_modules():
        if isinstance(m, NDModule):
            m.mirror_nth_member(idx, src.get_submodule(n))


@torch.no_grad()
def split_up_batched_ensemble(src: nn.Module):
    # determine the multiplicity
    for n, m in src.named_modules():
        if isinstance(m, NDModule):
            n_groups = m.n
            break

    # copy the scaffold
    batchmod = [deepcopy(src) for i in range(n_groups)]

    # fill the scaffolds
    for name, m in src.named_modules():
        if isinstance(m, NDModule):
            for i in range(n_groups):
                # extract all the individual submodules
                set_submodule(batchmod[i], name, m.get_nth_member(i))

    return batchmod


@torch.no_grad()
def feed_batch_model_image_multiple(img: torch.Tensor, n_models: int):
    # img : [batch, channels, H, W]
    return img.repeat(1, n_models, 1, 1)


@torch.no_grad()
def feed_batch_model_vector_data(vd: torch.Tensor, n_models: int):
    # vd : [batch, features]
    return vd.unsqueeze(0).expand(n_models, -1, -1)


def batch_output_for_ce_loss_computation(output: torch.Tensor):
    # output is assumed to have come out of LinearND layer
    # bring it to [N, C,...,] from [n, b, out]
    return torch.permute(output, (1,2,0)).contiguous()


def batch_target_for_ce_loss_computation(target: torch.Tensor, n_models: int):
    # targets are those for single model for the batch:
    # [N], need [N, n_models] to match the signature
    return target.unsqueeze(1).repeat(1, n_models)


class ModelBatchedCELoss(nn.CrossEntropyLoss):
    def __init__(self, n_models, *args, reduction='none', **kwargs):
        self.n_models = n_models
        # use 'none' reduction, as we have to sum over the number of models for independent gradients
        super(ModelBatchedCELoss, self).__init__(reduction=reduction, *args, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = batch_output_for_ce_loss_computation(input)
        if target.ndim < 2:
            target = batch_target_for_ce_loss_computation(target, self.n_models)
        loss = super(ModelBatchedCELoss, self).forward(input, target)
        assert len(loss.shape) == 2
        # average over batch and all dimensions behind (there should be none), sum over n dimensions for independent 
        if self.reduction == 'none':
            return loss
        else:
            print(f"Warning: this thing is better off without reduction")
            return torch.sum(torch.mean(loss, dim=1), dim=0)


class ModelBatchedAccuracy(nn.Module):
    def __init__(self, n_models):
        super(ModelBatchedAccuracy, self).__init__()
        self.register_full_backward_hook(_raise_exception)
        self.n_models = n_models

    def forward(self, y_hat, y):
        '''

        :param y_hat:
        [M, B, O]
        :param y:
        [B]
        :return:
        '''
        # print(y_hat.shape)
        # print(torch.argmax(y_hat, dim=(-1)))
        y_hat_pred = torch.argmax(y_hat, dim=(-1))  # [M, B]
        # print(y_hat_pred, y, y.unsqueeze(0).expand(self.n_models, y.shape[0]))
        return torch.sum(y_hat_pred == y.unsqueeze(0).expand(self.n_models, y.shape[0]), dim=-1) / y.shape[-1]
