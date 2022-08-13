import sys
import unittest
from ddt import ddt, data, unpack
import src.dqn.dqn as target

'''
the code below unittest SimulateOneStep
'''
@ddt
class testSimulateOneStep(unittest.TestCase):

    def setUp(self):
        def fakeTransition(state, action):
            return [3,7]

        def fakeReward(state,acton,next_state):
            return 67

        def fakeTerminalCheck(state):
            return False
        self.fakeTransition = fakeTransition
        self.fakeReward = fakeReward
        self.fakeTerminalCheck = fakeTerminalCheck
        
    @data(([2,1],1, 67,[3,7],False),
          ([5,2],0, 67,[3,7],False))
    @unpack
    def test(self,  state, action,
             expect_reward, expect_state, expect_terminal):
        simulator = target.SimulateOneStep(self.fakeTransition,self.fakeReward,self.fakeTerminalCheck)
        sim_reward, sim_next_state, sim_terminal = simulator(state, action)
        self.assertEqual(sim_reward, expect_reward)
        self.assertListEqual(sim_next_state, expect_state)
        self.assertEqual(sim_terminal,expect_terminal)


'''
the code below unittest TrainOneStep
'''
@ddt
class testTrainOneStep(unittest.TestCase):

    def setUp(self):
        class fakemodel1(object):
            def __call__(self, input1):
                return target.torch.as_tensor([[1, 2]], dtype=target.torch.float32)
        def fakePolicyEgreedy(Q,e):
            return 1
        class FakeSimulateOneStep(object):
            def __call__(self, state, action):
                return 3,[1,2],False
        def fakeSampleFromMemory(size, memory):
            return 2
        class FakeLearnFromMemory(object):
            def __call__(self, model, target_model, minibatch):
                return "model","target_model"

        self.memory =target.deque(maxlen=3)
        self.test_model = fakemodel1()
        self.test_target_model = fakemodel1()
        self.fake_SimulatorOneStep = FakeSimulateOneStep()
        self.fake_LearnfromMemory = FakeLearnFromMemory()
        self.fakePolicyEgreedy = fakePolicyEgreedy
        self.fakeSampleFromMemory = fakeSampleFromMemory
        
                
    
    @data((5,[0,1], 0.6,"model","target_model", [[0,1], 1, 3, [1,2], False], [1,2], False))
    @unpack
    def test(self,minibatchSize,state,e,expect_model, expect_target_model, expect_memory, expect_next_state, expect_terminal):

        trainstep = target.TrainOneStep( self.fakePolicyEgreedy, self.fake_SimulatorOneStep, self.fakeSampleFromMemory,\
                                         self.fake_LearnfromMemory, minibatchSize)
        sim_model, sim_target_model, sim_memory,sim_next_state, sim_terminal = \
            trainstep(self.test_model, self.test_target_model, self.memory, state, e)

        self.assertEqual(sim_model, expect_model)
        self.assertEqual(sim_target_model, expect_target_model)
        self.assertListEqual(sim_memory[0][0], expect_memory[0])
        self.assertEqual(sim_memory[0][1], expect_memory[1])
        self.assertEqual(sim_memory[0][2], expect_memory[2])
        self.assertListEqual(sim_memory[0][3], expect_memory[3])
        self.assertEqual(sim_memory[0][4], expect_memory[4])
        self.assertListEqual(sim_next_state, expect_next_state)
        self.assertEqual(sim_terminal,expect_terminal)


'''
The code below unittest GetEpsilon
'''
@ddt
class testGetEpsilon(unittest.TestCase):
    @data((10, 7, 0.5, 7), (2, 0.1, 0.7, 1.4))
    @unpack
    def test(self, e, e_min, e_decay, expect_e):
        getEpsilon = target.GetEpsilon(e,e_min, e_decay)
        sim_e = getEpsilon()
        self.assertEqual(sim_e, expect_e)




# Run the unittest
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
