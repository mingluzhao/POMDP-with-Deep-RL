import sys
import unittest
from ddt import ddt, data, unpack
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
import storage as target


@ddt
class testStorage(unittest.TestCase):
    def setUp(self):
        # 4 steps before train
        self.rollout = target.RolloutStorage(4)

    # def initialize()
       

    @data((0,target.torch.tensor([1.0]),target.torch.tensor([0.0])))
    @unpack
    def testInsert(self,step,reward,mask):
        self.rollout.insert(step,reward,mask)
        # the first reward should still be 0 
        self.assertEqual(self.rollout.rewards.numpy()[0][0][0],0.0)
        # the second reward is set to 1 now
        self.assertEqual(self.rollout.rewards.numpy()[1][0][0],1.0)
        # the first mask for terminal should be 1
        self.assertEqual(self.rollout.masks.numpy()[0][0][0],1.0)
        # the second mask for temrinal should be 0
        self.assertEqual(self.rollout.masks.numpy()[1][0][0],0.0)

    @data((target.torch.tensor([2.0]),0.9))
    @unpack
    def testComputeTarget(self,nextValue,gamma):
        self.rollout.insert(0,target.torch.tensor([1.0]),target.torch.tensor([0.0]))
        self.rollout.computeTargets(nextValue,gamma)
        self.assertAlmostEqual(self.rollout.returns.numpy()[4][0][0],2.0,3)
        self.assertAlmostEqual(self.rollout.returns.numpy()[3][0][0],1.8,3)
        self.assertAlmostEqual(self.rollout.returns.numpy()[2][0][0],1.62,3)
        self.assertAlmostEqual(self.rollout.returns.numpy()[1][0][0],1.458,3)
        self.assertAlmostEqual(self.rollout.returns.numpy()[0][0][0],1.0,3)



if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
