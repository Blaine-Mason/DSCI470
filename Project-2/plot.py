import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval

data = pd.read_csv('/Users/blaine/Github/DSCI470/Project-2/data/TEST.csv',index_col=[0])
#print(literal_eval(data[str(1000)][2].replace(" ", ",")))

a = 1000
b = 5000
step = 1000
for i in range(2,11,1):
    plt.plot([i for i in range(a,b,step)], [data[str(num)][i] for num in range(a,b,step)], label=f"n={i}")
plt.title("Simplex Error w/ FB", fontsize=15)
plt.xlabel("Samples", fontsize=15)
plt.ylabel("Frobenius Norm of A - Ã", fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper right')
plt.show()
# %%
