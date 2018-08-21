
# coding: utf-8

# In[2]:
import numpy as np
import ActivationFunction

class NNN:
# In[3]:
    def __init__(self):
        self.network = {}
        #self.network['W1'] = np.ones((4,3))/10
        self.network['W1'] = np.random.normal(10,20,(4,3))#sigmoidを使う場合は大きすぎる平均0分散1
        #self.network['B1'] = np.ones(3)/10
        self.network['B1'] = np.random.normal(10,20,(1,3))
        #self.network['W2'] = np.ones((3,2))/10
        self.network['W2'] = np.random.normal(10,20,(3,2))
        #self.network['B2'] = np.ones(2)/10
        self.network['B2'] = np.random.normal(10,20,(1,2))
        #self.network['W3'] = np.ones((2,1))/10
        self.network['W3'] = np.random.normal(10,20,(2,1))
        #self.network['B3'] = np.ones(1)/10
        self.network['B3'] = np.random.normal(10,20,(1,1))
        print(self.network)

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
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['B1'], network['B2'], network['B3']
        a1 = np.dot(X,W1) + b1
        z1 = ActivationFunction.sigmoid_function(a1)
        a2 = np.dot(z1,W2) + b2
        z2 = ActivationFunction.sigmoid_function(a2)
        a3 = np.dot(z2,W3) + b3
        y = ActivationFunction.softmax(a3)

        return y


    # In[ ]:


    def Classification_two(y):
        if(y>0.5):
            return 1
        else :
            return 0


    # In[2]:


    def update(self,individual):

        self.network['W1'] = np.reshape(individual[:12],(4,3))
        self.network['B1'] = np.reshape(individual[12:15],(1,3))
        self.network['W2'] = np.reshape(individual[15:21],(3,2))
        self.network['B2'] = np.reshape(individual[21:23],(1,2))
        self.network['W3'] = np.reshape(individual[23:25],(2,1))
        self.network['B3'] = np.reshape(individual[25:26],(1,1))
        '''
        self.network['W1'] = np.reshape(network[:12],(4,3))
        self.network['B1'] = np.reshape(network[12:15],(1,3))
        self.network['W2'] = np.reshape(network[15:21],(3,2))
        self.network['B2'] = np.reshape(network[21:23],(1,2))
        self.network['W3'] = np.reshape(network[23:25],(2,1))
        self.network['B3'] = np.reshape(network[25:26],(1,1))
        '''
        '''
        print("W1 : ",end="")
        print(individual[:12])
        print("B1 : ",end="")
        print(individual[12:15])
        print("W2 : ",end="")
        print(individual[15:21])
        print("B2 : ",end="")
        print(individual[21:23])
        print("W3 : ",end="")
        print(individual[23:25])
        print("B3 : ",end="")
        print(individual[25:26])
        '''
        '''
        print("W1 : ")
        print(network[:12])
        print("B1 : ")
        print(network[12:15])
        print("W2 : ")
        print(network[15:21])
        print("B2 : ")
        print(network[21:24])
        print("W3 : ")
        print(network[23:26])
        print("B3 : ")
        print(network[25:26])
        '''
    # In[3]:


    def conclusion(self,observation,individual):
        X = observation
        y = NNN.forward(self.network,X)
        action = NNN.Classification_two(y)
        return action


    # 染色体：26
    # W1:0-12
    # reshahpe{4,3]
    # B1:13-16
