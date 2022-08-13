import sys
import unittest
from ddt import ddt, data, unpack
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
import src.dvrl.aesmc.state as target


@ddt
# This resample only resample particles, won't change what is inside each particle
class testResample(unittest.TestCase):
    @data((target.torch.tensor([[[1.0,2.0],[3.0,4.0]]]),target.torch.tensor([[0,0]]),target.torch.tensor([[[1.0,2.0],[1.0,2.0]]]).numpy()),\
          (target.torch.tensor([[[1.0,2.0],[3.0,4.0]]]),target.torch.tensor([[0,1]]),target.torch.tensor([[[1.0,2.0],[3.0,4.0]]]).numpy()),\
          (target.torch.tensor([[[1.0,2.0],[3.0,4.0]]]),target.torch.tensor([[1,0]]),target.torch.tensor([[[3.0,4.0],[1.0,2.0]]]).numpy()),\
          (target.torch.tensor([[[1.0,2.0],[3.0,4.0]]]),target.torch.tensor([[1,1]]),target.torch.tensor([[[3.0,4.0],[3.0,4.0]]]).numpy()))
    @unpack
    def testResample(self,value,index,expectedResample):
        outputResample = target.resample(value,index).numpy()
        self.assertEqual(outputResample.all(),expectedResample.all())


@ddt
class testState(unittest.TestCase):

    def setUp(self):
        self.st = target.State(x = target.torch.tensor([[[0.0,1.0]],[[2.0,3.0]]]),\
                               y = target.torch.tensor([[[4.0,5.0]],[[6.0,7.0]]]))

        self.stForExpand =target.State(x = target.torch.tensor([[[0.0,1.0]]]),\
                               y = target.torch.tensor([[[2.0,3.0]]]))

    def testIndex(self):
        # index 0 should return [[0,1]] for x, [[4,5]] for y
        indexSt = self.st.index_elements(0)
        self.assertEqual(indexSt.y.numpy().all(),target.np.array([[4.0,5.0]]).all())
        self.assertEqual(indexSt.x.numpy().all(),target.np.array([[0.0,1.0]]).all())

    def testUnqueezeExpand(self):

        self.stForExpand.unsqueeze_and_expand_all_(dim=2,size=4)
        # unittest if it is corrected expanded to size 4
        self.assertEqual(self.stForExpand.x.size()[2],4)
        self.assertEqual(self.stForExpand.y.size()[2],4)
        # unittest if it is unsqueezed from 3 to 4 dimension
        self.assertEqual(len(self.stForExpand.x.size()),4)
        self.assertEqual(len(self.stForExpand.y.size()),4)
        # unittest if all expanded values are the same
        for i in range(len(self.stForExpand.x.size())-1):
            self.assertEqual(self.stForExpand.x[0][0][i].numpy().all(),self.stForExpand.x[0][0][i+1].numpy().all())

        for i in range(len(self.stForExpand.y.size())-1):
            self.assertEqual(self.stForExpand.y[0][0][i].numpy().all(),self.stForExpand.y[0][0][i+1].numpy().all())

    def testMultiplyEach(self):
        #only multiply each value in x by 50, return a new state that only has x*50
        newSt_x = self.stForExpand.multiply_each(target.torch.tensor([[50.0]]),"x")
        self.assertEqual(newSt_x.x.numpy()[0][0][1],50.0)
        self.assertEqual(newSt_x.x.numpy()[0][0][0],0.0)
        #only multiply each value in y by 50, return a new state that only has y*50
        newSt_y = self.stForExpand.multiply_each(target.torch.tensor([[50.0]]),"y")
        self.assertEqual(newSt_y.y.numpy()[0][0][1],150.0)
        self.assertEqual(newSt_y.y.numpy()[0][0][0],100.0)


    def testUpdate(self):
        # unittest if newSt is correctly updated and appended to self.stForExpand
        newSt = target.State(z=target.torch.tensor([[[4.0,5.0]]]))
        self.stForExpand.update(newSt)
        self.assertEqual(self.stForExpand.z.numpy()[0][0][0],4.0)

    
        
     




if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)




        
