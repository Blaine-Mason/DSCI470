
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import fblib as fblib

#samples = 50000
percent = .9
a = 1000
b = 2000
step = 1000
plts = True
fb = True
shape = "simplex"
avg = 1
n = 2
df = pd.DataFrame(columns=np.arange(a,b,step), index=[f"{n}"])

if fb:
    error, samples = fblib.get_data(n, percent, a, b, step, plts, fb, shape, avg)
    temp = np.array(list(zip(error, samples)))
    j = 0
    for i in range(a, b, step):
        df.loc[str(n)][i] = temp[j]
        j += 1
else:
    error = fblib.get_data(n, percent, a, b, step, plts, fb, shape, avg)
    j = 0
    for i in range(a, b, step):
        df.loc[str(n)][i] = error[j]
        j += 1
print(df)