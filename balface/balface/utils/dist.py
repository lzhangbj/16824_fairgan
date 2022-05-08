import random
import numpy as np
import torch
import torch.distributed as dist

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
                  for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def sync_tensor_across_gpus(t):
    # t needs to have dim 0 for troch.cat below.
    # if not, you need to prepare it.
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t)  # this works with nccl backend when tensors need to be on gpu.
   # for gloo and mpi backends, tensors need to be on cpu. also this works single machine with
   # multiple   gpus. for multiple nodes, you should use dist.all_gather_multigpu. both have the
   # same definition... see [here](https://pytorch.org/docs/stable/distributed.html).
   #  somewhere in the same page, it was mentioned that dist.all_gather_multigpu is more for
   # multi-nodes. still dont see the benefit of all_gather_multigpu. the provided working case in
   # the doc is  vague...
    return torch.cat(gather_t_tensor, dim=0)