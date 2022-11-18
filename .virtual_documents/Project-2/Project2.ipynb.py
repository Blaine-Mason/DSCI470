import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("data/cbb.csv")


data['POSTSEASON'] = data['POSTSEASON'].fillna(0)
data['SEED'] = data['SEED'].fillna(0)


yearly = [data[data["YEAR"] == x] for x in range(2013, 2022)]


data['CONF'].unique() 


temp = yearly[0]
confs = ["ACC", "B10", "B12", "P12", "A10", "IVY","SEC"]
temp = temp[temp['CONF'].isin(confs)]


temp


fig = plt.figure(figsize=(12,12))

sns.set_style('darkgrid')
sns.heatmap(temp.corr(), annot=True, square=True, cmap='coolwarm')


temp[temp['POSTSEASON'] get_ipython().getoutput("= 0]")



