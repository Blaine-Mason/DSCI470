from consolemenu import *
from consolemenu.items import *
import numpy as np
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

# Create the menu
menu = ConsoleMenu("Data Tool", "When plotting, if done successfully the script will crash and return to the menu.  This means things worked. The library I am using is no longer maintained, but does what I need.")

# Create some items
def test(b):
    a = int(input("Enter a lower bound for samples(e.g. 1000): "))
    b = int(input("Enter a upper bound for samples(e.g. 5000): "))
    step = int(input("Enter a step value(e.g. 100): "))
    avg = int(input("Enter the number of trials to average over(e.g. 10): "))
    filename = input("Enter a file name(do not include file ext): ")
    if a > b or step > b or a < 0 or b < 0:
        print("Use your brain, follow directions")
        Screen().input('Press [Enter] to continue')
        return
    fb = b
    df = pd.DataFrame(columns=np.arange(a,b,step), index=["2", "3", "4", "5", "6", "7",
            "8", "9", "10"])
    for n in tqdm(np.arange(2,11,1)):
        print(f"n={n}")
        error = fblib.get_data(n, percent, a, b, step, plts, fb, shape, avg)
        j = 0
        for i in range(a,b,step):
            df.loc[str(n)][i] = error[j]
            j += 1
    df.to_csv(f'/Users/blaine/Github/DSCI470/Project-2/data/{filename}.csv')
    Screen().input('Press [Enter] to continue')


def plts():
    filename = input("Enter a file name of data(do not include file ext): ")
    filename_fig = input("Enter a file to save figure as(do not include file ext): ")
    data = pd.read_csv(f'/Users/blaine/Github/DSCI470/Project-2/data/{filename}.csv',index_col=[0])
    #print(literal_eval(data[str(1000)][2].replace(" ", ",")))

    a = int(input("Enter the lower bound for your samples: "))
    b = int(input("Enter the upper bound for samples: "))
    step = int(input("Enter the step value: "))
    for i in range(2,11,1):
        plt.plot([i for i in range(a,b,step)], [data[str(num)][i] for num in range(a,b,step)], label=f"n={i}")
    plt.title("Simplex Error w/ FB", fontsize=15)
    plt.xlabel("Samples", fontsize=15)
    plt.ylabel("Frobenius Norm of A - Ã", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper right')
    plt.savefig(f"{filename_fig}.png")

# A FunctionItem runs a Python function when selected
get_data = FunctionItem("Generate a Dataset(w/FB)", test, args=[True])
get_data_wo = FunctionItem("Generate a Dataset(wo/FB)", test, args=[False])
gen_plot = FunctionItem("Plot Data", plts)

# Once we're done creating them, we just add the items to the menu
menu.append_item(get_data)
menu.append_item(get_data_wo)
menu.append_item(gen_plot)

# Finally, we call show to show the menu and allow the user to interact
menu.show()