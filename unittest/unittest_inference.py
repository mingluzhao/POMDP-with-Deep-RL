import sys
import unittest
from ddt import ddt, data, unpack
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
import aesmc.inference as target


@ddt
# No need to further unittest cuz this calls an already tested function
class testGetNormWeights(unittest.TestCase):
    @data((target.torch.tensor([[1.0,2.0,3.0],[2.0,3.0,5.0]]),target.torch.tensor([[0.0900, 0.2447, 0.6652],[0.0420, 0.1142, 0.8438]])))
    @unpack
    def test_normal_case(self,inputTensor,expectedTensor):
        outputTensor = target.getNormWeights(inputTensor).numpy()
        self.assertAlmostEqual(outputTensor[0][0],expectedTensor.numpy()[0][0],3)
        self.assertAlmostEqual(outputTensor[1][0],expectedTensor.numpy()[1][0],3)

        

@ddt
class testGetCumulativeWeights(unittest.TestCase):
    @data((target.torch.tensor([[0.0900, 0.2447, 0.6652],[0.0420, 0.1142, 0.8438]]),\
           target.torch.tensor([[0.0900, 0.3348, 1.0000],[0.0420, 0.1562, 1.0000]])))
    @unpack
    def test_normal_case(self,inputWeights,expectedWeights):
        outputWeights = target.getCumulativeWeights(inputWeights).numpy()
        self.assertAlmostEqual(outputWeights[0][0],expectedWeights.numpy()[0][0],3)
        self.assertAlmostEqual(outputWeights[1][0],expectedWeights.numpy()[1][0],3)


class testGetPos(unittest.TestCase):

    def setUp(self):
        self.sampleMethod = lambda x: target.np.ones([x,1])/10

    def test_normal_case(self):
        # batch, num_particles, sample method
        batch_size = 1
        num_particles =3
        pos = target.getPos(batch_size,num_particles,self.sampleMethod)
        self.assertAlmostEqual(pos[0][0],0.0333,3)
        self.assertAlmostEqual(pos[0][1],0.36667,3)
        self.assertAlmostEqual(pos[0][2],0.7,3)

    # Test when there are more than one batch
    def test_two_batch(self):
        batch_size = 2
        num_particles =3
        pos = target.getPos(batch_size,num_particles,self.sampleMethod)
        self.assertAlmostEqual(pos[0][0],0.0333,3)
        self.assertAlmostEqual(pos[0][1],0.36667,3)
        self.assertAlmostEqual(pos[0][2],0.7,3)
        self.assertAlmostEqual(pos[1][0],0.0333,3)
        self.assertAlmostEqual(pos[1][1],0.36667,3)
        self.assertAlmostEqual(pos[1][2],0.7,3)


@ddt
class testSampleAncestralIndex(unittest.TestCase):

    def setUp(self):
        self.sampleMethod_small = lambda x: target.np.ones([x,1])/10
        self.sampleMethod_large = lambda x: target.np.ones([x,1])/1.25

    @data((target.torch.tensor([[1.0,2.0,3.0]]),target.torch.tensor([[0,2,2]])))
    @unpack
    # Output will be 0, 2, 2 given the noise = 0.1
    def test_normal_case(self,inputWeights,expectedIndex):
        outputIndex = target.sample_ancestral_index(inputWeights,self.sampleMethod_small)
        self.assertEqual(outputIndex[0][0],expectedIndex[0][0])
        self.assertEqual(outputIndex[0][1],expectedIndex[0][1])
        self.assertEqual(outputIndex[0][2],expectedIndex[0][2])

    @data((target.torch.tensor([[1.0,2.0,3.0]]),target.torch.tensor([[1,2,2]])))
    @unpack
    # Output will be 1, 2, 2 given the noise = 0.8
    def test_large_noise(self,inputWeights,expectedIndex):
        outputIndex = target.sample_ancestral_index(inputWeights,self.sampleMethod_large)
        self.assertEqual(outputIndex[0][0],expectedIndex[0][0])
        self.assertEqual(outputIndex[0][1],expectedIndex[0][1])
        self.assertEqual(outputIndex[0][2],expectedIndex[0][2])
        
    
    




if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
