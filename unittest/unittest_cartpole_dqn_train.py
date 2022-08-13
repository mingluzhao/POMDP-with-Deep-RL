import sys
import unittest
from ddt import ddt, data, unpack
import exec.dqn_cartpole as target

'''
The code below unittest Train
'''


@ddt
class testTrain(unittest.TestCase):

    def setUp(self):
        class fakemodel2(object):
            def __init__(self):
                self.s_dict = 0

            def __call__(self, input1):
                return [1, 2]

            def state_dict(self):
                return self.s_dict

            def load_state_dict(self, state_dict):
                self.s_dict = state_dict

        class FakeEpsilon(object):
            def __call__(self):
                return 20

        class FakeTrainOneStep(object):
            def __call__(self, model, target_model, memory, state, e):
                return model, target_model, [1000], [1, 2], False

        class FakeSimulator(object):
            def get_initial_state(self):
                return [0, 0]

        self.fake_TrainOneStep = FakeTrainOneStep()
        self.fake_getE = FakeEpsilon()
        self.fake_Simulator = FakeSimulator()
        self.fake_model = fakemodel2()
        self.fake_target = fakemodel2()

    @data((20, 30,50,[0.2]),
          (10, 7, 5,[1000]))
    @unpack
    def test(self,maxSteps, maxEpisodes, update_freq,memory):
        fakeTrain = target.Train(self.fake_TrainOneStep, maxSteps, maxEpisodes, self.fake_getE, update_freq)
        sim_model = fakeTrain(self.fake_model, self.fake_target, memory,self.fake_Simulator)
        self.assertListEqual(sim_model(1), [1,2])
        self.assertEqual(sim_model.s_dict, 0)


# Run the unittest
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)