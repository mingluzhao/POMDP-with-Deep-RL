from . import math
import time
import numpy as np
import torch


def getNormWeights(log_weight):
    return np.exp(math.lognormexp(
        log_weight.cpu().detach(),
        dim=1
    ))

def getCumulativeWeights(normalized_weights):
    
    # np.ndarray [batch_size, num_particles]
    cumulative_weights = np.cumsum(normalized_weights, axis=1)

    # trick to prevent numerical issues
    cumulative_weights = cumulative_weights / np.max(
        cumulative_weights.numpy(),
        axis=1,
        keepdims=True
    )

    return cumulative_weights


    

def getPos(batch_size,num_particles,sample_method):

    noise = sample_method(batch_size)
    
    pos = (noise + np.arange(0, num_particles)) / num_particles

    return pos




def sample_ancestral_index(log_weight,sampleNoiseMethod): # ancestral_log_weight
   
    """Sample ancestral index using systematic resampling.

    input:
        log_weight: log of unnormalized weights, Tensor/Variable
            [batch_size, num_particles]
    output:
        zero-indexed ancestral index: LongTensor/Variable
            [batch_size, num_particles]
    """

    device = log_weight.device
    assert(torch.sum(log_weight != log_weight) == 0)
    batch_size, num_particles = log_weight.size()
    indices = np.zeros([batch_size, num_particles])

    normalized_weights = getNormWeights(log_weight)

    cumulative_weights = getCumulativeWeights(normalized_weights)

    pos = getPos(batch_size,num_particles,sampleNoiseMethod)
    # compute weight
    
    # sample(weight,position)

    for batch in range(batch_size):
        indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])

    temp = torch.from_numpy(indices).long().to(device)
  

    return temp
