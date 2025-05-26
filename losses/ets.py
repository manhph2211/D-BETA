import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    recv_tensor = torch.zeros_like(tensor)
    ops = [
        dist.P2POp(dist.isend, tensor, to_rank, group=group),
        dist.P2POp(dist.irecv, recv_tensor, from_rank, group=group)
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return recv_tensor


def neighbour_exchange_bidir(left_rank, right_rank, tensor_left, tensor_right, group=None):
    recv_left = torch.zeros_like(tensor_right)
    recv_right = torch.zeros_like(tensor_left)
    ops = [
        dist.P2POp(dist.isend, tensor_left, left_rank, group=group),
        dist.P2POp(dist.isend, tensor_right, right_rank, group=group),
        dist.P2POp(dist.irecv, recv_left, left_rank, group=group),
        dist.P2POp(dist.irecv, recv_right, right_rank, group=group),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    return recv_right, recv_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output)
        return None, None, None, grad_input


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_left, tensor_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_left, tensor_right, group)

    @staticmethod
    def backward(ctx, grad_left, grad_right):
        grads = NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, grad_right, grad_left)
        return None, None, None, *grads


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_left, tensor_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_left, tensor_right)


class ETSLoss(nn.Module):
    def __init__(self, cache_labels=False, rank=0, world_size=1, bidir=True, use_horovod=False):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.bidir = bidir
        self.use_horovod = use_horovod

    def get_ground_truth(self, labels: torch.Tensor, num_logits: int) -> torch.Tensor:
        gt = -torch.ones((num_logits, num_logits), device=labels.device, dtype=labels.dtype)
        diag = torch.arange(num_logits, device=labels.device)
        gt[diag, diag] = 2 * labels - 1
        return gt

    def get_logits(self, image_feats, text_feats, logit_scale, bias=None):
        logits = logit_scale * image_feats @ text_feats.T
        return logits + bias if bias is not None else logits

    def _loss(self, image_feats, text_feats, logit_scale, labels, bias=None):
        logits = self.get_logits(image_feats, text_feats, logit_scale, bias)
        gt = self.get_ground_truth(labels, image_feats.size(0))
        return -F.logsigmoid(gt * logits).mean()

    def forward(self, image_feats, text_feats, labels, logit_scale=torch.tensor(1.0), bias=None, output_dict=False):
        loss = self._loss(image_feats, text_feats, logit_scale, labels, bias)

        if self.world_size > 1:
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size

            if self.bidir:
                left_feat = right_feat = text_feats
                num_rounds, has_extra = divmod(self.world_size - 1, 2)
                for _ in range(num_rounds):
                    left_feat, right_feat = neighbour_exchange_bidir_with_grad(left_rank, right_rank, left_feat, right_feat)
                    for feat in (left_feat, right_feat):
                        loss += self._loss(image_feats, feat, logit_scale, labels, bias)
                if has_extra:
                    extra_feat = neighbour_exchange_with_grad(left_rank, right_rank, right_feat)
                    loss += self._loss(image_feats, extra_feat, logit_scale, labels, bias)
            else:
                to_right = text_feats
                for _ in range(self.world_size - 1):
                    from_left = neighbour_exchange_with_grad(left_rank, right_rank, to_right)
                    loss += self._loss(image_feats, from_left, logit_scale, labels, bias)
                    to_right = from_left

        return {"ets_loss": loss} if output_dict else loss
