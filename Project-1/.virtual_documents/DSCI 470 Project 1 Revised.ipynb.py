import numpy as np

def Cost(w,b,x,y):
    # We take in a w,b value and a vector of features x and the target y#
    
    C=.5*np.sum(((w*x+b)-y)**2)
    
    return C
def f(x,w,b):
    return w*x + b


def Gradient_descent(X,y,w,b,alpha=.005,iterations=1000):
    w_i = w
    b_i = b
    n = float(len(x))
    previous_cost = None
    for it in range(iterations):
        current_cost = Cost(w_i, b_i, X, y)
        if previous_cost and abs(previous_cost-current_cost)<= 1e-6:
            break
        previous_cost = current_cost
        
        # Calculating the gradients
        pd_w = np.sum((f(x, w_i, b_i)-y)*x)
        pd_b = np.sum((f(x, w_i, b_i)-y))
         
        # Updating weights and bias
        w_i = w_i - (alpha * pd_w)
        b_i = b_i - (alpha * pd_b)
        if it % 10:
            print(f"w: {w_i}, b: {b_i}, cost: {current_cost}")
    return w_i, b_i, current_cost
w,b,cost=Gradient_descent(x,y,1,4.8,.001,100)
print(f"w: {w}, b: {b}, cost: {cost}")


import pandas as pd
import matplotlib.pyplot as plt

url="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data" 
_df=pd.read_csv(url, header=None)
_df[1:10]
d = _df.values


x=_df[23].to_numpy()
x=np.reshape(x, (len(x), 1))



y=_df[24].to_numpy()
y=np.reshape(y, (len(y), 1))

plt.scatter(x,y)
plt.xlabel("City");
plt.ylabel("Highway");



# imports and setup 
import pandas as pd
import scipy as sc
import numpy as np
import seaborn as sns
sns.set()

# imports ski-kit modules
from sklearn import tree, svm, metrics
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
#get_ipython().run_line_magic("matplotlib", " notebook")
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
get_ipython().run_line_magic("matplotlib", " inline  ")
plt.rcParams['figure.figsize'] = (10, 6) 


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here


# your code goes here
