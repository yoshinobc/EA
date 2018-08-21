class NN:
    import numpy as np
    def __init__(self,x,v,theta,av):
        self.x = x
        self.v = v
        self.theta = theta
        self.av = av
    def init_network():
        network = {}
        network['W1'] = np.ones((4,3))/10
        network['B1'] = np.ones(3)/10
        network['W2'] = np.ones((3,2))/10
        network['B2'] = np.ones(2)/10
        network['W3'] = np.ones((2,1))/10
        network['B3'] = np.ones(1)/10

        return network

    def create_input_neuron(observation):
        X = observation

        return X

    def forward(network,x,v,theta,av):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        a1 = np.dot(X,W1) + B1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + B2
        z2 = sigmoid(a2)
        a3 = np.dot(z2,W3) + B3
        y = ActivityFunction.softmax_function(a3)

        return y

    def Classification_two(y):
        if(y>0.5):
            return 1
        else :
            return 0

    def update(individual):
        network['W1'] = np.reshape(individual[:11],(4,3))
        network['B1'] = np.reshape(individual[11:14],(1,3))
        network['W2'] = np.reshape(individual[14:21],(3,2))
        network['B2'] = np.reshape(individual[21:23],(1,2))
        network['W3'] = np.reshape(individual[23:25],(2,1))
        network['B3'] = np.reshape(individual[25:26],(1,1))

    def conclusion(observation,individual):
        network = init_network()
        X = create_input_neuron(observation)
        y = forward(network,X)
        action = Classification_two(y)
        update(individual)
        return action
