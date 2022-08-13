import sys
import unittest
from ddt import ddt, data, unpack
import cartpole_dqn_torch_final as target

       
'''
The code below test policy_egreedy
'''
@ddt
class testpolicyEgreedy(unittest.TestCase):
    @data((target.T.as_tensor([[2,1]],dtype=target.T.float32),0),(target.T.as_tensor([[1,2]],dtype=target.T.float32),1))
    @unpack
    def test(self,Q,expectedAction):
        # test only when e=0, meaning no exploration
        calculatedAction = target.policyEgreedy(Q,0)
        # should always choose to perform action 0 based on the fake_getQ
        self.assertEqual(calculatedAction, expectedAction)
      
'''
The code below test SamplefromMemory. Random.sample() uniformly samples from the given list
'''

@ddt
class testSamplefromMemory(unittest.TestCase):

    def setUp(self):
        self.memory = target.deque(maxlen=3)
        self.memory.append(1)
        self.memory.append(2)
        self.memory.append(3)
        
    @data((1,0.333,0.333,0.333))
    @unpack
    def test(self,batch_size,expected_freq_1, expected_freq_2,expected_freq_3):
        
        sample = [target.sampleFromMemory(batch_size,self.memory)[0] for trials in range(100000)]
        calculated_freq_1 = sample.count(1)/100000
        calculated_freq_2 = sample.count(2)/100000
        calculated_freq_3 = sample.count(3)/100000
        self.assertAlmostEqual(calculated_freq_1,expected_freq_1,places=2)
        self.assertAlmostEqual(calculated_freq_2,expected_freq_2,places=2)
        self.assertAlmostEqual(calculated_freq_3,expected_freq_3,places=2)

'''
The code below tests LearnFromOneEpisode  (the update of Q-value based on bellman equation )
'''
@ddt
class testLearnfromOneSample(unittest.TestCase):

    def setUp(self):
        class fakemodel(object):
            def __call__(self,input1):
                return target.T.as_tensor([[1,2]],dtype=target.T.float32)
        
        self.test_model  = fakemodel()
        self.test_episode_normal = [target.T.as_tensor([[ 0.04816886,  0.02680717,  0.02389789, -0.03381444]]), 
                         1, 1.0, target.T.as_tensor([[-0.03134213,  0.19668025, -0.0006552 , -0.30045238]]), False]

    
    @data((0.9,[1,2],[1,2.8]))
    @unpack
    def test(self,gamma,expected_pred,expected_target):
        learnfromonesample = target.LearnFromOneSample(gamma)
        calculated_pred, calculated_target= learnfromonesample(self.test_model,self.test_model,self.test_episode_normal)
        self.assertEqual(calculated_target[0],expected_target[0])
        self.assertEqual(calculated_target[1],expected_target[1])
        self.assertEqual(calculated_pred[0],expected_pred[0])
        self.assertEqual(calculated_pred[1],expected_pred[1])
    
    
           
'''
The code below test LearnFromMemory
'''
@ddt
class testLearnFromMemory(unittest.TestCase):

    def setUp(self):
        def Fake_learnbackprop(model,target_model,minibatch,learnFromOneEpisode):
            return 6
        self.fake_learnBackProp = Fake_learnbackprop
        
    @data((0.25,2,5,5,[2],0.25),\
          (0.25,2,5,5,[],0))
    @unpack
    # We expect to see 1/4 of the times that model is changed from 5 to 6
    # We expect to see 0 of the times that model is changed from 5 to 6 if the minibatch is empty
    def test(self,train_freq,fake_learnfromoneepisode,fake_model,fake_target_model,fake_minibatch,expected_freq):
        
        learnfrommemory = target.LearnFromMemory(fake_learnfromoneepisode,train_freq,self.fake_learnBackProp)
        model_list = [learnfrommemory(fake_model,fake_target_model,fake_minibatch)[0] for trials in range(100000)]
        actual_freq = model_list.count(6)/100000
        self.assertAlmostEqual(actual_freq, expected_freq, places=2)



# Run the test
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
        
       
        
