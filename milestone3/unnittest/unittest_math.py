import sys
import unittest
from ddt import ddt, data, unpack
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
import aesmc.math as target



@ddt
class testLogSumExp(unittest.TestCase):
    @data((target.torch.tensor([[1.0,2.0,3.0],[2.0,3.0,4.0]]),target.torch.tensor([2.3133, 3.3133, 4.3133])))
    @unpack
    # just a normal case 
    def test_normalcase(self,inputTensor,expectedTensor):
        outputTensor = target.logsumexp(inputTensor,dim=0).numpy()
        self.assertAlmostEqual(outputTensor[0],expectedTensor.numpy()[0],3)
        self.assertAlmostEqual(outputTensor[1],expectedTensor.numpy()[1],3)
        self.assertAlmostEqual(outputTensor[2],expectedTensor.numpy()[2],3)
    @data((target.torch.tensor([[0.0,0.0,0.0]]),target.torch.tensor([0.0,0.0,0.0])))
    @unpack
    # test when the first dimension = 1, should return itself with reduced dimension
    def test_one_dimension(self,inputTensor,expectedTensor):
        outputTensor = target.logsumexp(inputTensor,dim=0).numpy()
        self.assertAlmostEqual(outputTensor[0],expectedTensor.numpy()[0],3)
        self.assertAlmostEqual(outputTensor[1],expectedTensor.numpy()[1],3)
        self.assertAlmostEqual(outputTensor[2],expectedTensor.numpy()[2],3)

    @data((target.torch.tensor([[-1.0],[-2.0]]),target.torch.tensor([-0.6867])))
    @unpack
    # test when the input is negative
    def test_negative(self,inputTensor,expectedTensor):
        outputTensor = target.logsumexp(inputTensor,dim=0).numpy()
        self.assertAlmostEqual(outputTensor[0],expectedTensor.numpy()[0],3)
       
    


@ddt
class testLogNormExp(unittest.TestCase):
    @data((target.torch.tensor([[1.0,2.0,3.0],[2.0,3.0,4.0]]),target.torch.tensor([[-1.3133, -1.3133, -1.3133],[-0.3133, -0.3133, -0.3133]])))
    @unpack
    # just a normal case
    def test_normalcase(self,inputTensor,expectedTensor):
        outputTensor = target.lognormexp(inputTensor,dim=0).numpy()
        self.assertAlmostEqual(outputTensor[0][0],expectedTensor.numpy()[0][0],3)
        self.assertAlmostEqual(outputTensor[1][0],expectedTensor.numpy()[1][0],3)

    @data((target.torch.tensor([[1.0,2.0,3.0]]),target.torch.tensor([[0.0,0.0,0.0]])))
    @unpack
    # test when the first dimension = 1, should return a tensor filled with 0.0
    def test_one_dimension(self,inputTensor,expectedTensor):
        outputTensor = target.lognormexp(inputTensor,dim=0).numpy()
        self.assertAlmostEqual(outputTensor[0][0],expectedTensor.numpy()[0][0],3)
        self.assertAlmostEqual(outputTensor[0][1],expectedTensor.numpy()[0][1],3)

    @data((target.torch.tensor([[-1.0],[-2.0]]),target.torch.tensor([[-0.3133],[-1.3133]])))
    @unpack
    # test when the input is negative
    def test_negative(self,inputTensor,expectedTensor):
        outputTensor = target.lognormexp(inputTensor,dim=0).numpy()
        self.assertAlmostEqual(outputTensor[0][0],expectedTensor.numpy()[0][0],3)
        self.assertAlmostEqual(outputTensor[1][0],expectedTensor.numpy()[1][0],3)
    
    
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
