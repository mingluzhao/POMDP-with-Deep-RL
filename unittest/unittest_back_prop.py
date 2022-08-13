import sys
import unittest
from ddt import ddt, data, unpack
import src.dqn.dqn as target


# The code below unittest learnbackprop (a little bit complicated)

class Build_Test_Model(target.nn.Module):

    # This fake model only has one weight. It has the following formula:
    #    output = input * weight(initialized to be 2)
    #    loss = (target-output)^2 = output^2 -2*output*target + target*2
    #    Gradient = dloss/dw = dloss/doutput * doutput/dw = (2 * output-2*target) * input = 2*(output-target)*input
    #    weight = weight - gradient*lr
    
    
    def __init__(self):
        super(Build_Test_Model, self).__init__()
        self.nn_layers = target.nn.ModuleList([target.nn.Linear(1, 1, bias=False)])
        # learning rate is fixed as 0.1
        # Use SGD instead of Adam because it is easier to calculate gradient update
        self.optimizer = target.optim.SGD(self.parameters(), lr= 0.1)
        self.loss = target.nn.MSELoss()
        target.torch.nn.init.constant_(self.nn_layers[0].weight, 2)
        self.device = target.torch.device('cuda:0' if target.torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = target.torch.tensor(state, dtype=target.torch.float32).to(self.device)
        state = self.nn_layers[0](state)
        return state


class Fake_LearnFromOneSample(object):
    # This fake function always return prediction = 4, target = 7 if the above fake model is passed in 
    def __call__(self, model, target_model, episode):
        state = target.torch.tensor([2], dtype=target.torch.float32)
        pred = model(state)
        result_target = target.torch.tensor([7], dtype=target.torch.float32)
        return pred, result_target


test_model = Build_Test_Model()
test_target_model = test_model
test_learn = Fake_LearnFromOneSample()
# The minibatch is not important as the model always return prediction =4, target = 7
test_minibatch = [5]

# begin to unittest if the back propogration is correctly calculated
@ddt
class testlearnbackprop(unittest.TestCase):
    # The new weight should be 3.2 after one iteration of training: 2-2*(4-7)*2*lr = 2-(-1.2) = 3.2. Formula for the update is shown above
    @data((test_model,test_target_model,test_learn,test_minibatch,3.2))
    @unpack
    def test(self, test_model,test_target_model,test_learn,test_minibatch,expected_weight):
        test_model = target.learnbackprop(test_model,test_target_model,test_minibatch,test_learn)
        calculated_weight = [param for param in test_model.parameters()][0]
        self.assertAlmostEqual(calculated_weight, expected_weight, places = 3)
    




if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
