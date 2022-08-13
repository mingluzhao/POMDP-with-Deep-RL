import torch.nn as nn
import torch.nn.functional as F


class Categorical(nn.Module):
    def __init__(self, numInputs, numOutputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(numInputs, numOutputs)

    def forward(self, x):
        x = self.linear(x)
        return x

    def sample(self, x, deterministic):
        x = self(x)
        probs = F.softmax(x, dim=-1)
        if deterministic is False:
            action = probs.multinomial(1)
        else:
            action = probs.max(1, keepdim=True)[1]
        return action

    def logprobs_and_entropy(self, x, actions):
        x = self(x)
        logProbs = F.log_softmax(x, dim=-1)
        probs = F.softmax(x, dim=-1)
        actionLogProbs = logProbs.gather(1, actions)
        distEntropy = -(logProbs * probs).sum(-1).mean()
        return actionLogProbs, distEntropy
