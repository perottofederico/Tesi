import math
import pickle
import torch
from accelerate import Accelerator
from torch.autograd import Function 
from torch.utils.data.distributed import DistributedSampler


# Initialize the accelerator
accelerator = Accelerator()


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation using Accelerator.
    """

    @staticmethod
    def forward(ctx, x):
        # Get world size from accelerator
        world_size = accelerator.num_processes
        
        # Use Accelerator's gather method to collect tensors from all devices
        gathered = accelerator.gather(x)
        
        # Split the gathered tensor to match torch.distributed behavior
        output = list(torch.chunk(gathered, world_size, dim=0))
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        # Stack gradients
        all_gradients = torch.stack(grads)
        
        # Use Accelerator to perform all_reduce equivalent
        # We handle this by gathering all gradients and then broadcasting the sum
        all_gradients_gathered = accelerator.gather(all_gradients)
        all_gradients_sum = all_gradients_gathered.sum(dim=0)
        
        # Return the gradient for the current process
        process_idx = accelerator.process_index
        return all_gradients_sum[process_idx]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors using Accelerator.
    Graph remains connected for backward grad computation.
    """
    # Get world size from accelerator
    world_size = accelerator.num_processes
    
    # There is no need for reduction in the single-process case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)
    return torch.cat(tensor_all, dim=0)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors using Accelerator.
    Note: This operation has no gradient.
    """
    return accelerator.gather(tensor)


def ddp_allgather(input):
    """Gather input from all processes with support for different batch sizes."""
    # Get the size of the input
    size = torch.tensor(input.shape[0], device=accelerator.device)
    
    # Gather sizes from all processes
    gathered_sizes = accelerator.gather(size.unsqueeze(0)).squeeze()
    max_size = gathered_sizes.max().item()
    
    # Pad the input if necessary
    padding_size = max_size - size
    if padding_size > 0:
        padding_tensor = torch.zeros(padding_size, *input.shape[1:], device=accelerator.device)
        padded_input = torch.cat((input, padding_tensor), dim=0)
    else:
        padded_input = input
    
    # Gather all padded inputs
    gathered_inputs = accelerator.gather(padded_input)
    
    # Extract actual data based on original sizes
    output = []
    offset = 0
    for s in gathered_sizes:
        s = s.item()
        output.append(gathered_inputs[offset:offset + s])
        offset += max_size  # Skip to the next process's data
    
    # Concatenate all outputs
    return torch.cat(output, dim=0)


class DistributedSampler_wopadding(DistributedSampler):
    """A modified DistributedSampler that doesn't add padding samples."""

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Remove tail of data to make it evenly divisible if drop_last is True
        if self.drop_last:
            indices = indices[:self.total_size]

        # Subsample based on rank and world size
        indices = indices[self.rank:len(indices):self.num_replicas]

        return iter(indices)




