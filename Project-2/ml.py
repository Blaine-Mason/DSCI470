import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import fblib as fblib
from tqdm import tqdm

#samples = 50000
percent = .9
a = 1000
b = 5000
step = 1000
plts = False
fb = True
shape = "simplex"
avg = 1

df = pd.DataFrame(columns=np.arange(a,b,step), index=["2", "3", "4", "5", "6", "7",
                  "8", "9", "10"])
for n in tqdm(np.arange(2,11,1)):
    print(f"n={n}")
    error = fblib.get_data(n, percent, a, b, step, plts, fb, shape, avg)
    j = 0
    for i in range(a,b,step):
        df.loc[str(n)][i] = error[j]
        j += 1
df.to_csv('/Users/blaine/Github/DSCI470/Project-2/data/TEST.csv')