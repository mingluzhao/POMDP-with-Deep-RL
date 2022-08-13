import numpy as np
import random


class WumpusEnv(object):
    def __init__(self, observationAccu):
        self.observationAccu = observationAccu
        self.terminal = False
        self.agentPlaceSpace = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (2, 2)]
        self.wumpusPlaceSpace = [(2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (2, 2)]
        self.stateSpace = [(agentPlace, wumpusPlace) for agentPlace in self.agentPlaceSpace for wumpusPlace in
                           self.wumpusPlaceSpace if agentPlace != wumpusPlace] + [((0, 0), (0, -1))]
        self.actionSpace = ['MoveUp', 'MoveDown', 'MoveLeft', 'MoveRight', 'ShootUp', 'ShootRight', 'ShootDown',
                            'ShootLeft']
        self.observationSpace = [('Stench',), ('',)]
        moveTable = {'MoveUp': (0, 1), 'MoveDown': (0, -1), 'MoveLeft': (-1, 0), 'MoveRight': (1, 0)}

        def moveFunction(s, a, sPrime):
            if s == ((0, 0), (0, -1)):
                return 1 * (sPrime == ((0, 0), (0, -1)))
            agentPlace, wumpusPlace = s
            x, y = agentPlace
            dx, dy = moveTable[a]
            agentPlacePrime = (x + dx, y + dy)
            if agentPlacePrime not in self.agentPlaceSpace:
                agentPlacePrime = agentPlace
            wumpusPlacePrime = wumpusPlace
            if agentPlacePrime == wumpusPlacePrime:
                return 1 * (sPrime == ((0, 0), (0, -1)))
            return 1 * ((agentPlacePrime, wumpusPlacePrime) == sPrime)

        shootFunction = lambda s, a, sPrime: 1 * (sPrime == ((0, 0), (0, -1)))
        actionTransition = {'MoveUp': moveFunction, 'MoveDown': moveFunction, 'MoveLeft': moveFunction,
                            'MoveRight': moveFunction,
                            'ShootUp': shootFunction, 'ShootRight': shootFunction, 'ShootDown': shootFunction,
                            'ShootLeft': shootFunction}
        transitionFunc = lambda s, a, sPrime: actionTransition[a](s, a, sPrime)

        transitionMatrix = np.array(
            [[[transitionFunc(s, a, sPrime) for sPrime in self.stateSpace] for a in self.actionSpace] for s in
             self.stateSpace])

        movingCost = -5  # TODO: used movingCost=-2 in dvrl
        victoryReward = 100
        losingPenalty = -100

        def shootingReward(s, a, sPrime):
            if s == ((0, 0), (0, -1)):
                return 0
            agentPlace, wumpusPlace = s
            x, y = agentPlace
            xWumpus, yWumpus = wumpusPlace
            dx, dy = (xWumpus - x, yWumpus - y)
            if a == 'ShootUp':
                return victoryReward * ((dx, dy) == (0, 1)) + losingPenalty * ((dx, dy) != (0, 1))
            if a == 'ShootRight':
                return victoryReward * ((dx, dy) == (1, 0)) + losingPenalty * ((dx, dy) != (1, 0))
            if a == 'ShootDown':
                return victoryReward * ((dx, dy) == (0, -1)) + losingPenalty * ((dx, dy) != (0, -1))
            if a == 'ShootLeft':
                return victoryReward * ((dx, dy) == (-1, 0)) + losingPenalty * ((dx, dy) != (-1, 0))

        movingReward = lambda s, a, sPrime: movingCost * (sPrime != ((0, 0), (0, -1))) + losingPenalty * (
                    sPrime == ((0, 0), (0, -1))) * (s != ((0, 0), (0, -1)))
        rewardTable = {'MoveUp': movingReward, 'MoveDown': movingReward, 'MoveLeft': movingReward,
                       'MoveRight': movingReward,
                       'ShootUp': shootingReward, 'ShootRight': shootingReward, 'ShootDown': shootingReward,
                       'ShootLeft': shootingReward}
        rewardFunc = lambda s, a, sPrime: rewardTable[a](s, a, sPrime)

        rewardMatrix = np.array(
            [[[rewardFunc(s, a, sPrime) for sPrime in self.stateSpace] for a in self.actionSpace] for s in
             self.stateSpace])

        # observationAccuracy=modelDf.index.get_level_values('observationAccuracy')[0]

        def observationFunc(sPrime, a, o):
            agentPlace, wumpusPlace = sPrime
            if agentPlace == (0, 0) or agentPlace in self.wumpusPlaceSpace:
                return 1 * (o == ('',))
            x, y = agentPlace
            xWumpus, yWumpus = wumpusPlace
            adjacency = (abs(xWumpus - x) + abs(yWumpus - y)) == 1
            if adjacency:
                return (o == ('Stench',)) * self.observationAccu + (o == ('',)) * (1 - self.observationAccu)
            else:
                return (o == ('Stench',)) * (1 - self.observationAccu) + (o == ('',)) * self.observationAccu

        observationMatrix = np.array(
            [[[observationFunc(sPrime, a, o) for o in self.observationSpace] for a in self.actionSpace] for sPrime in
             self.stateSpace])

        self.transitionFunction = lambda s, a, sPrime: transitionMatrix[s, a, sPrime]
        self.rewardFunction = lambda s, a, sPrime: rewardMatrix[s, a, sPrime]
        self.observationFunction = lambda sPrime, a, o: observationMatrix[sPrime, a, o]

    def getInitialState(self):
        possibleInitial = [0, 1, 2, 3, 4, 5]
        return random.sample(possibleInitial, 1)[0]

    def transition(self, state, action):
        # this is deterministic transition
        # number of possible next state is 49
        for i in range(49):
            if self.transitionFunction(state, action, i) != 0:
                if i == 48:
                    self.terminal = True
                else:
                    self.terminal = False
                return i

    def reward(self, state, action, sprime):
        return self.rewardFunction(state, action, sprime)

    def observation(self, sprime, a):
        prob_stench = self.observationFunction(sprime, a, 0)
        prob_nostench = self.observationFunction(sprime, a, 1)
        return random.choices([0, 1], [prob_stench, prob_nostench])[0]

    def isterminal(self, state):
        return self.terminal
