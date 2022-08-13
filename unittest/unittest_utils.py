import sys
import unittest
from ddt import ddt, data, unpack
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
import utils as target


@ddt
class testToOneHot(unittest.TestCase):
    # unittest single action and multiple actions
    @data((3,target.torch.tensor([[2],[2]]),target.torch.tensor([[0.0,0.0,1.0],[0.0,0.0,1.0]])),
          (3,target.torch.tensor([1]),target.torch.tensor([[0.0,1.0,0.0]])))
    @unpack
    def testToOneHot(self,actionDims,inputAction,expectedAction):

        outputAction = target.toOneHot(actionDims,inputAction)
        self.assertEqual(outputAction.numpy().all(),expectedAction.numpy().all())
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
