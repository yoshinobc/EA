
# coding: utf-8

import numpy as np
import ActivationFunction

class BNN:

    def __init__(self,input_size, hidden1_size,hidden2 outputu_size, weight_init_std=0.01) :

        self.network = {}
        self.network['W1'] = weight_init_std * np.random.randn(input_size,hidden1_size)
        self.network['B1'] = np.zeros(hidden1_size)
        self.network['W2'] = weight_init_std * np.random.randn(hidden1_size,hidden2_size)
        self.network['B2'] = np.zeros(hidden2_size)
        self.network['W3'] = weight_init_std * np.random.randn(hidden2_size,output_size)
        self.network['B3'] = np.zeros(output_size)                
    # In[1]:


    def create_input_neuron(self,observation):
        X = observation

        return X


    # In[4]:


    def forward(network,X):

        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['B1'], network['B2'], network['B3']
        a1 = np.dot(X,W1) + b1
        z1 = ActivationFunction.sigmoid_function(a1)
        a2 = np.dot(z1,W2) + b2
        z2 = ActivationFunction.sigmoid_function(a2)
        a3 = np.dot(z2,W3) + b3
        y = ActivationFunction.sigmoid_function(a3)

        return y


    # In[ ]:


    def Classification_two(y):
        if(y>0.5):
            return 1
        else :
            return 0


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
        self.network['W1'] = np.reshape(individual[:24],(4,6))
        self.network['B1'] = np.reshape(individual[24:30],(1,6))
        self.network['W2'] = np.reshape(individual[30:48],(6,3))
        self.network['B2'] = np.reshape(individual[48:51],(1,3))
        self.network['W3'] = np.reshape(individual[51:54],(3,1))
        self.network['B3'] = np.reshape(individual[54:55],(1,1))
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


    def conclusion(self,observation,individual,network):
        X = observation
        y = NNN.forward(network,X)
        action = NNN.Classification_two(y)
        return action
