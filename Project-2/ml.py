import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import fblib as fblib

samples = 10000
n = 2
percent = 1
a = 1000
b = 2000
step = 1000
plts = False
fb = False
shape = "simplex"
avg = 1
n = 2
df = pd.DataFrame(columns=np.arange(a,b,step), index=[f"{n}"])
error = fblib.lls(n, percent, samples, plts, fb)