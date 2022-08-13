import torch


def toOneHot(actionDim, actions):
    actionsOnehotDim = list(actions.size())
    actionsOnehotDim[-1] = actionDim

    actions = actions.view(-1, 1).long()
    actionOnehot = torch.FloatTensor(actions.size(0), actionDim)

    actionOnehot.zero_()
    actionOnehot.scatter_(1, actions, 1)
    actionOnehot.view(*actionsOnehotDim)

    return actionOnehot
