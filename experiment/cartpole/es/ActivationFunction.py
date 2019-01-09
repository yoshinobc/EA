
# coding: utf-8

# In[8]:


class ActivationFunction:
    def __init__(self,x):
        self.x = x
        import numpy as np


# In[3]:


def step_function(x):
    if x>0:
        return 1
    else :
        return 0


# In[4]:


def sigmoid_function(x):
    import numpy as np
    return 1/(1+np.exp(x))


# In[5]:


def relu_function(x):
    import numpy as np
    return np.maximum(0,x)


# In[6]:


def identity_function(x):
    return x


# In[7]:


def softmax(x):
    import numpy as np
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
