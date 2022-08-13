import random


class EasyWumpusEnv(object):
    def __init__(self,observationAccu):
        self.stateSpace=[(agentLocation, wumpusLocation) for agentLocation in [0, 1] for wumpusLocation in [0, 1, 2]]+[(0,-1)]
        self.actionSpace=[0,1,2,3]
        
        self.observationSpace=[0,1]
        self.observationAccu = observationAccu
        def transitionFunction(s, a, sPrime):
    
            state_dict = {0:(0,0),1:(0,1),2:(0,2),3:(1,0),4:(1,1),5:(1,2),6:(0,-1)}
            if s==6:
                return 1*(sPrime==6)
            if a in [2, 3]:
                return 1*(sPrime==6)
            if a==0:
                return 1*(sPrime==s)
            if a==1:
                if sPrime==6:
                    return 0
                agentLocation, wumpusLocation=state_dict[s]
                agentLocationPrime, wumpusLocationPrime=state_dict[sPrime]
                return (agentLocation==0 and agentLocationPrime==1 and wumpusLocation==wumpusLocationPrime)+\
                       (agentLocation== 1and agentLocationPrime==0 and wumpusLocation==wumpusLocationPrime)
    
        def rewardFunctionFull(s, a, sPrime, movingCost, victoryReward, losingPenalty):
            state_dict = {0:(0,0),1:(0,1),2:(0,2),3:(1,0),4:(1,1),5:(1,2),6:(0,-1)}
            if s==6:
                return 0
            if a not in [2, 3]:
                return movingCost
            agentLocation, wumpusLocation=state_dict[s]
            if a==2:
                return (wumpusLocation==0)*victoryReward+(wumpusLocation!=0)*losingPenalty
            if a==3:
                if agentLocation==0:
                    return (wumpusLocation==1)*victoryReward+(wumpusLocation!=1)*losingPenalty
                if agentLocation==1:
                    return (wumpusLocation==2)*victoryReward+(wumpusLocation!=2)*losingPenalty
                

        def observationFunctionFull(sPrime, a, o, observationAccuracy):
            state_dict = {0:(0,0),1:(0,1),2:(0,2),3:(1,0),4:(1,1),5:(1,2),6:(0,-1)}
            if sPrime==6:
                return 1*(o==('',))
            agentLocation, wumpusLocation=state_dict[sPrime]
            if agentLocation==0:
                if wumpusLocation in [0,1]:  
                    return observationAccuracy*(o==0)+(1-observationAccuracy)*(o==1)
                else:
                    return (1-observationAccuracy)*(o==0)+observationAccuracy*(o==1)
            if agentLocation==1:
                if wumpusLocation in [0,2]:
                    return observationAccuracy*(o==0)+(1-observationAccuracy)*(o==1)
                else:
                    return (1-observationAccuracy)*(o==0)+observationAccuracy*(o==1)
            return 0

        movingCost=-1
        victoryReward=100
        losingPenalty=-100
        self.rewardFunction=lambda s, a, sPrime: rewardFunctionFull(s, a, sPrime, movingCost, victoryReward, losingPenalty)
        self.observationFunction=lambda sPrime, a, o: observationFunctionFull(sPrime, a, o, self.observationAccu)
        self.transitionFunction = lambda s,a,sPrime:transitionFunction(s, a, sPrime)

    def getInitialState(self):
        possibleInitial = [0,1,2]
        return random.sample(possibleInitial,1)[0]
     
    
    def transition(self,state,action):
        # this is deterministic transition
        # number of possible next state is 49
        for i in range(7):
            if self.transitionFunction(state,action,i)!=0:
                if i ==6:
                    self.terminal = True
                else:
                    self.terminal = False
                return i
        
    def reward(self,state,action,sprime):
        return self.rewardFunction(state,action,sprime)
    
    def observation(self,sprime,a):
        prob_stench = self.observationFunction(sprime,a,0)
        prob_nostench = self.observationFunction(sprime,a,1)
        return random.choices([0,1],[prob_stench,prob_nostench])[0]

    def isterminal(self,state):
        return self.terminal
       
        
    
    



