
# coding: utf-8

# In[2]:
import numpy as np

class NNN:
# In[3]:
    def __init__(self):
        self.network = {}
        """
        #self.network['W1'] = np.ones((4,3))/10
        self.network['W1'] = np.random.normal(0,1,(4,3))#sigmoidを使う場合は大きすぎる平均0分散1
        #self.network['B1'] = np.ones(3)/10
        self.network['B1'] = np.random.normal(0,1,(1,3))
        #self.network['W2'] = np.ones((3,2))/10
        self.network['W2'] = np.random.normal(0,1,(3,2))
        #self.network['B2'] = np.ones(2)/10
        self.network['B2'] = np.random.normal(0,1,(1,2))
        #self.network['W3'] = np.ones((2,1))/10
        self.network['W3'] = np.random.normal(0,1,(2,1))
        #self.network['B3'] = np.ones(1)/10
        self.network['B3'] = np.random.normal(0,1,(1,1))
        print(self.network)
        """
    '''
    def init_network(self):
        import numpy as np
        self.network = {}
        self.network['W1'] = np.ones((4,3))/10
        self.network['B1'] = np.ones(3)/10
        self.network['W2'] = np.ones((3,2))/10
        self.network['B2'] = np.ones(2)/10
        self.network['W3'] = np.ones((2,1))/10
        self.network['B3'] = np.ones(1)/10
    '''

    # In[1]:


    def create_input_neuron(self,observation):
        X = observation

        return X

    # In[4]:


    def forward(network,X):
        def sigmoid2(x):
            return (1-np.exp(-x))/(1+np.exp(-x))

        def sigmoid(x):
            import scipy.special
            return scipy.special.expit(x)
        """
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['B1'], network['B2'], network['B3']
        a1 = np.dot(X,W1) + b1
        z1 = ActivationFunction.sigmoid_function(a1)
        a2 = np.dot(z1,W2) + b2
        z2 = ActivationFunction.sigmoid_function(a2)
        a3 = np.dot(z2,W3) + b3
        y = ActivationFunction.sigmoid_function(a3)
        """
        #W1, W2, W3 = network['W1'], network['W2'], network['W3']
        #b1, b2, b3 = network['B1'], network['B2'], network['B3']
        W1, W2 = network['W1'], network['W2']
        b1, b2 = network['B1'], network['B2']
        a1 = np.dot(X,W1) + b1
        z1 = a1
        a2 = np.dot(z1,W2) + b2
        z2 = a2
        #a3 = np.dot(z2,W3) + b3
        #y = (sigmoid(a3) - 0.5) * 2
        y = (sigmoid(z2) - 0.5) * 2
        return y[0]


    # In[ ]:

#nanがどこからきてるのか確認



    # In[2]:


    def update(self,individual):
        """
        self.network['W1'] = np.reshape(individual[:12],(4,3))
        self.network['B1'] = np.reshape(individual[12:15],(1,3))
        self.network['W2'] = np.reshape(individual[15:21],(3,2))
        self.network['B2'] = np.reshape(individual[21:23],(1,2))
        self.network['W3'] = np.reshape(individual[23:25],(2,1))
        self.network['B3'] = np.reshape(individual[25:26],(1,1))
        """
        #self.network['W1'] = np.reshape(individual[:384],(24,16))
        #self.network['B1'] = np.reshape(individual[384:400],(1,16))
        #self.network['W2'] = np.reshape(individual[400:528],(16,8))
        #self.network['B2'] = np.reshape(individual[528:536],(1,8))
        #self.network['W3'] = np.reshape(individual[536:568],(8,4))
        #self.network['B3'] = np.reshape(individual[568:572],(1,4))
        self.network['W1'] = np.reshape(individual[:240],(24,10))
        self.network['B1'] = np.reshape(individual[240:250],(1,10))
        self.network['W2'] = np.reshape(individual[250:290],(10,4))
        self.network['B2'] = np.reshape(individual[290:294],(1,4))

        """
        self.network['W1'] = np.reshape(individual[:24],(4,6))
        self.network['B1'] = np.reshape(individual[24:30],(1,6))
        self.network['W2'] = np.reshape(individual[30:60],(6,5))
        self.network['B2'] = np.reshape(individual[60:65],(1,5))
        self.network['W3'] = np.reshape(individual[65:80],(5,3))
        self.network['B3'] = np.reshape(individual[80:83],(1,3))
        self.network['W4'] = np.reshape(individual[83:86],(3,1))
        self.network['B4'] = np.reshape(individual[86:87],(1,1))
        """
        return self.network
        '''
        self.network['W1'] = np.reshape(network[:12],(4,3))
        self.network['B1'] = np.reshape(network[12:15],(1,3))
        self.network['W2'] = np.reshape(network[15:21],(3,2))
        self.network['B2'] = np.reshape(network[21:23],(1,2))
        self.network['W3'] = np.reshape(network[23:25],(2,1))
        self.network['B3'] = np.reshape(network[25:26],(1,1))
        '''

    # In[3]:


    def conclusion(self,observation,network):
        X = observation
        y = NNN.forward(network,X)
        action = y
        return action
