import sys
import unittest
from ddt import ddt, data, unpack
import os
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
import aesmc.random_variable as target



# there is only one function in Bernoulli class to test: logpdf
@ddt
class testBernoulli(unittest.TestCase):
    
    def setUp(self):
        # batch, num_particles, observation_dim
        self.bernoulli = target.MultivariateIndependentPseudobernoulli(target.torch.tensor([[[0.3,0.3,0.3],[0.3,0.3,0.3]]]))

    @data((target.torch.tensor([[[0.2,0.2,0.4],[0.2,0.15,0.4]]]),target.torch.tensor([[-1.7479,-1.7055]]).numpy()))
    @unpack
    def testLogPdf(self,inputValue,expectedLogPdf):
        outputLogPdf = self.bernoulli.logpdf(inputValue,1,2).numpy()
        self.assertAlmostEqual(outputLogPdf[0][0],expectedLogPdf[0][0],3)
        self.assertAlmostEqual(outputLogPdf[0][1],expectedLogPdf[0][1],3)





@ddt
class testNormal(unittest.TestCase):

    # TODO sd
    
    def setUp(self):
        self.normal = target.MultivariateIndependentNormal(target.torch.tensor([[[1.0]]]),target.torch.tensor([[[0.0]]]))
        self.multiNormal = target.MultivariateIndependentNormal(target.torch.tensor([[[1.0,2.0]]]),target.torch.tensor([[[0.0,0.0]]]))
        self.normalWithSd = target.MultivariateIndependentNormal(target.torch.tensor([[[1.0]]]),target.torch.tensor([[[0.5]]]))
    # it should always return 1.0 because mean = 1, sd = 0
    def testSample(self):
        sampledValue = self.normal.sample(1,1)
        self.assertEqual(sampledValue[0][0][0],1.0)

    def testSampleRepara(self):
        sampledValue = self.normal.sample_reparameterized(1,1)
        self.assertEqual(sampledValue[0][0][0],1.0)
        
    # test multivariate normal sample
    def testMultiSample(self):
        sampledValue = self.multiNormal.sample(1,1)
        self.assertEqual(sampledValue[0][0][0],1.0)
        self.assertEqual(sampledValue[0][0][1],2.0)

    def testMultiSampleRepara(self):
        sampledValue = self.multiNormal.sample_reparameterized(1,1)
        self.assertEqual(sampledValue[0][0][0],1.0)
        self.assertEqual(sampledValue[0][0][1],2.0)

    # testLogpdf method
    @data((target.torch.tensor([[[2.0]]]),target.torch.tensor([[-1.5724]]).numpy()))
    @unpack
    def testLogPdf(self,inputValue,expectedLogPdf):
        outputLogPdf = self.normalWithSd.logpdf(inputValue,1,1).numpy()
        self.assertAlmostEqual(outputLogPdf[0][0],expectedLogPdf[0][0],3)



        

@ddt
# state random variable is a collection of random variable objects
# Its function will use these random variable objects's methods by iterating over them
class testStateRandomVariable(unittest.TestCase):

    def setUp(self):
        # stateRV contains z 
        self.stateRV = target.StateRandomVariable(z =target.MultivariateIndependentNormal(target.torch.tensor([[[1.0]]]),target.torch.tensor([[[0.0]]])))
        # this stateRV with_sd tests logpdf function
        # Without standard deviation it will return Nan
        self.stateRV_with_sd = target.StateRandomVariable(z =target.MultivariateIndependentNormal(target.torch.tensor([[[1.0]]]),\
                                                                                                  target.torch.tensor([[[0.5]]])))

        # this stateRV contains two random variables z and h
        self.stateRV_zh = target.StateRandomVariable(z =target.MultivariateIndependentNormal(target.torch.tensor([[[1.0]]]),target.torch.tensor([[[0.0]]])),
                                                     h =target.MultivariateIndependentNormal(target.torch.tensor([[[2.0]]]),target.torch.tensor([[[0.0]]])) )

    # here the input could be a dictionary of many things, we need to find common key that has the name "z"
    @data((target.st.State(z=target.torch.tensor([[[2.0]]]),h=target.torch.tensor([[[2.5]]])),["z"]))
    @unpack
    def testFindSingleCommonKey(self,inputDict,expectedKey):
        outputKey = self.stateRV._find_common_keys(inputDict)
        self.assertEqual(outputKey,expectedKey)

    @data((target.st.State(z=target.torch.tensor([[[2.0]]]),h=target.torch.tensor([[[2.5]]])),["z","h"]))
    @unpack
    def testFindMultiCommonKeys(self,inputDict,expectedKey):
        outputKey_1 = self.stateRV_zh._find_common_keys(inputDict)[0]
        outputKey_2 = self.stateRV_zh._find_common_keys(inputDict)[1]
        self.assertIn(outputKey_1,expectedKey)
        self.assertIn(outputKey_2,expectedKey)

    # this will sample every random variable and return a dictionary contains name : sampled value
    # Here the dict should contains z: [[[1.0]]] 
    def testSampleRepara(self):
        sampledDict = self.stateRV.sample_reparameterized(1,1)
        self.assertEqual(sampledDict.z.numpy()[0][0][0],1.0)

    def testSample(self):
        sampledDict = self.stateRV.sample(1,1)
        self.assertEqual(sampledDict.z.numpy()[0][0][0],1.0)

    # this will sample two random variables: z and h
    def testSampleMultiRepara(self):
        sampledDict = self.stateRV_zh.sample_reparameterized(1,1)
        self.assertEqual(sampledDict.z.numpy()[0][0][0],1.0)
        self.assertEqual(sampledDict.h.numpy()[0][0][0],2.0)

    def testSampleMulti(self):
        sampledDict = self.stateRV_zh.sample(1,1)
        self.assertEqual(sampledDict.z.numpy()[0][0][0],1.0)
        self.assertEqual(sampledDict.h.numpy()[0][0][0],2.0)


        
    # this will test LogPdf
    # Single RV case is enough
    @data((target.st.State(z=target.torch.tensor([[[2.0]]]),h=target.torch.tensor([[[2.5]]])),target.torch.tensor([[-1.5724]]).numpy()))
    @unpack
    def testLogPdf(self,inputDict,expectedLogPdf):
        outputLogPdf = self.stateRV_with_sd.logpdf(inputDict,1,1).numpy()
        self.assertAlmostEqual(outputLogPdf[0][0],expectedLogPdf[0][0],3)
   
        
        
        

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
