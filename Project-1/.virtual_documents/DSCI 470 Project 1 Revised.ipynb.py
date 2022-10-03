import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn as m
url="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data" 
_df=pd.read_csv(url, header=None)
x,y = m.datasets.make_wave(n_samples=100)
x = x.reshape(1,-1)[0]
def Cost(w,b,x,y):
    # We take in a w,b value and a vector of features x and the target y#
    
    C=.5*np.sum(((w*x+b)-y)**2)
    
    return C
def f(x,w,b):
    return w*x + b

def gradientDescent(x, y, theta, learn_rate, N, n_iter):
    loss_i = np.zeros(n_iter)
    for i in range(n_iter):
        w = theta[0]
        b = theta[1]
        yhat = w*x+b
        loss = np.sum((yhat-y)** 2)/(2)
        loss_i[i] = loss
        #print("i:get_ipython().run_line_magic("d,", " loss: %f\" % (i, loss))")

        gradient_w = np.dot(x,(yhat-y).T)
        gradient_b = np.sum((yhat-y))
        w = w - learn_rate*gradient_w
        b = b - learn_rate*gradient_b
        theta = [w,b]
    return w,b,loss_i
regression = []
for alpha in np.arange(.001, .00001, -.00001):
    w,b, loss = gradientDescent(x.T,y,np.zeros(2),alpha, x.shape[0],100)
    regression.append(loss[-1])


plt.plot([i for i in np.arange(.001, .00001, -.00001)], regression)


plt.scatter(x,y)
plt.xlabel("x");
plt.ylabel("y");
plt.plot(x, [w*a + b for a in x], 'r--')


# imports and setup 
import pandas as pd
import scipy as sc
import numpy as np
import seaborn as sns
sns.set()

# imports ski-kit modules
from sklearn import tree, svm, metrics
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
#get_ipython().run_line_magic("matplotlib", " notebook")
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
get_ipython().run_line_magic("matplotlib", " inline  ")
plt.rcParams['figure.figsize'] = (15, 11) 


df = pd.concat([pd.read_csv("train1.csv"),pd.read_csv("train2.csv")])
df.head()


parsed_df = df[['Acres', 'Deck', 'GaragCap', 'Latitude', 'Longitude', 'DaysOnMkt', 'LstPrice', 'Patio', 'PkgSpacs', 'PropType', 'SoldPrice', 'Taxes', 'TotBed', 'TotBth', 'TotSqf', 'YearBlt']]


parsed_df.head()


parsed_df['Acres'] = parsed_df['Acres'].astype(float)
parsed_df['Deck'] = parsed_df['Deck'].astype(int)
parsed_df['GaragCap'] = parsed_df['GaragCap'].astype(int)
parsed_df['Latitude'] = parsed_df['Latitude'].astype(float)
parsed_df['Longitude'] = parsed_df['Longitude'].astype(float)
parsed_df['DaysOnMkt'] = parsed_df['DaysOnMkt'].astype(float)
parsed_df['LstPrice'] = parsed_df['LstPrice'].astype(int)
parsed_df['Patio'] = parsed_df['Patio'].astype(int)
parsed_df['PkgSpacs'] = parsed_df['PkgSpacs'].astype(int)
parsed_df['PropType'].replace(['Condo', 'Townhouse','Single Family'],
                        [0, 0, 1], inplace=True)
parsed_df['SoldPrice'] = parsed_df['SoldPrice'].astype(int)
parsed_df['Taxes'] = parsed_df['Taxes'].astype(int)
parsed_df['TotBed'] = parsed_df['TotBed'].astype(float)
parsed_df['TotBth'] = parsed_df['TotBth'].astype(float)
parsed_df['TotSqf'] = parsed_df['TotSqf'].str.replace(',', '').astype(int)
parsed_df['YearBlt'] = parsed_df['YearBlt'].astype(int)
parsed_df = parsed_df.drop(parsed_df[parsed_df['Taxes'] >= 90000].index)
parsed_df = parsed_df.drop(parsed_df[parsed_df['Longitude'] == 0].index)


parsed_df.dropna(inplace=True)


df['PropType'].value_counts().plot(kind='bar');


corr = parsed_df.corr()
parsed_df.corr()


sns.heatmap(corr, cmap="Blues", annot=True, fmt='.2g')


scatter_df = parsed_df[['Acres', 'LstPrice', 'SoldPrice', 'Taxes', 'TotBed', 'PropType', 'TotBth', 'TotSqf', 'YearBlt']]
sns.pairplot(scatter_df, hue="PropType");



pip install plotly


import plotly.express as px
geo = parsed_df[['Latitude', 'Longitude', 'SoldPrice']]
fig = px.scatter_mapbox(geo, lat="Latitude", lon="Longitude", color="SoldPrice",
                  color_continuous_scale=px.colors.cyclical.IceFire, zoom=11.5)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


import mglearn
knn_data = parsed_df[["Taxes", "SoldPrice"]]
X_ = knn_data["Taxes"].values.reshape(1, -1).T
y_ = knn_data["SoldPrice"].values.reshape(1, -1).T
n=1
model = KNeighborsRegressor(n_neighbors=n)


X_train, X_test, y_train, y_test = train_test_split(X_, y_, random_state=0)
model.fit(X_train, y_train)


print("Test set R^2: {:.2f}".format(model.score(X_test, y_test)))


test_accuracy = []
training_accuracy = []
# try n_neighbors from 1 to 10 
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors=n_neighbors) 
    clf.fit(X_train, y_train)
    # record training set accuracy 
    training_accuracy.append(clf.score(X_train, y_train)) 
    # record generalization accuracy 
    test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


fig, axes = plt.subplots(1, 5, figsize=(20, 5))
rmse_val = [] #to store rmse values for different k
line = np.linspace(0, 25000, 323)[:, np.newaxis]
for n, ax in zip([1, 3, 5, 9,10], axes):
    model = KNeighborsRegressor(n_neighbors=n)
    y = model.fit(X_train, y_train).predict(line)
    error = np.sqrt(mean_squared_error(y_,y)) #calculate rmse
    rmse_val.append(model.score(X_train, y_train)) #store rmse values
    ax.plot(line, model.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n, model.score(X_train, y_train), model.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    axes[0].legend(["Model predictions", "Training data/target",
                    "Test data/target"], loc="best")
#     plt.scatter(X_, y_, color="darkorange", label="data")
#     plt.plot(T, y, color="navy", label="prediction")
#     plt.axis("tight")
#     plt.legend()
#     plt.title("KNeighborsRegressor (k = get_ipython().run_line_magic("i)"", " % (n))")


reg_data = parsed_df[["SoldPrice", "Taxes"]]
X_ = reg_data["Taxes"].values.reshape(1, -1).T
y_ = reg_data["SoldPrice"].values.reshape(1, -1).T

reg = LinearRegression().fit(X_, y_)

y_pred = reg.predict(X_)

plt.scatter(X_, y_, color="black")
plt.plot(X_, y_pred, color="blue", linewidth=3)
plt.show()


reg.score(X_test, y_test)


y = parsed_df['LstPrice']
X = parsed_df[['Acres', 'Deck', 'GaragCap', 'Latitude', 'Longitude', 'DaysOnMkt', 'Patio', 'PkgSpacs', 'SoldPrice', 'Taxes', 'TotBed', 'TotBth', 'TotSqf', 'YearBlt']]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)
regressor.score(X_test, y_test)


parsed_df.insert(len(parsed_df.columns), "DiffPriceAbsolute", parsed_df['LstPrice'] - parsed_df['SoldPrice'])
parsed_df.insert(len(parsed_df.columns), "DiffPriceRelative", (parsed_df['SoldPrice'] - parsed_df['LstPrice'])/parsed_df['LstPrice'])


y = parsed_df['DiffPriceAbsolute']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)
regressor.score(X_test, y_test)


y = parsed_df['DiffPriceRelative']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)
print(regressor.score(X_test, y_test))


X = parsed_df[["SoldPrice"]]
y = parsed_df['DiffPriceRelative']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(regressor.score(X_test, y_test))


X = parsed_df[["SoldPrice"]]
y = parsed_df['DiffPriceAbsolute']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)
print(regressor.score(X_test, y_test))
plt.scatter(X, y, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.show()


# your code goes here


df2 = pd.read_csv("OnlineNewsPopularity.csv")


df2 = df2.drop(['url'], axis=1)
df2 = df2.drop([' timedelta'], axis=1)


X = df2[[' n_tokens_title', ' n_tokens_content', ' n_unique_tokens',
       ' n_non_stop_words', ' n_non_stop_unique_tokens', ' num_hrefs',
       ' num_self_hrefs', ' num_imgs', ' num_videos', ' average_token_length',
       ' num_keywords', ' data_channel_is_lifestyle',
       ' data_channel_is_entertainment', ' data_channel_is_bus',
       ' data_channel_is_socmed', ' data_channel_is_tech',
       ' data_channel_is_world', ' kw_min_min', ' kw_max_min', ' kw_avg_min',
       ' kw_min_max', ' kw_max_max', ' kw_avg_max', ' kw_min_avg',
       ' kw_max_avg', ' kw_avg_avg', ' self_reference_min_shares',
       ' self_reference_max_shares', ' self_reference_avg_sharess',
       ' weekday_is_monday', ' weekday_is_tuesday', ' weekday_is_wednesday',
       ' weekday_is_thursday', ' weekday_is_friday', ' weekday_is_saturday',
       ' weekday_is_sunday', ' is_weekend', ' LDA_00', ' LDA_01', ' LDA_02',
       ' LDA_03', ' LDA_04', ' global_subjectivity',
       ' global_sentiment_polarity', ' global_rate_positive_words',
       ' global_rate_negative_words', ' rate_positive_words',
       ' rate_negative_words', ' avg_positive_polarity',
       ' min_positive_polarity', ' max_positive_polarity',
       ' avg_negative_polarity', ' min_negative_polarity',
       ' max_negative_polarity', ' title_subjectivity',
       ' title_sentiment_polarity', ' abs_title_subjectivity',
       ' abs_title_sentiment_polarity',]].to_numpy()



med = np.median(df2[[' shares']].to_numpy())
df2[' shares'].loc[df2[' shares'] < med] = 0
df2[' shares'].loc[df2[' shares'] >= med] = 1
shares = df2[[' shares']].to_numpy()


print(f"Min: {min(shares)}, Max: {max(shares)}, Median: {np.median(shares)}")


X_train, X_test, y_train, y_test = train_test_split(X, shares)
Knn_class= KNeighborsClassifier(n_neighbors=1)
n_scores = cross_val_score(Knn_class, X_train, y_train.ravel(), scoring='accuracy', n_jobs=-1)


print("get_ipython().run_line_magic("0.2f", " accuracy with a standard deviation of %0.2f\" % (n_scores.mean(), n_scores.std()))")


linear_svm = svm.LinearSVC(max_iter=10000, dual=False).fit(X[0:8000,:], shares[0:8000].ravel())
n_scores = cross_val_score(linear_svm, X[0:8000,:], shares[0:8000].ravel(), scoring='accuracy', n_jobs=-1)


print("get_ipython().run_line_magic("0.2f", " accuracy with a standard deviation of %0.2f\" % (n_scores.mean(), n_scores.std()))")


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', class_weight='balanced', multi_class='multinomial', solver='lbfgs')
# define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X, shares, verbose=1, scoring='accuracy', cv=cv, n_jobs=-1)
# report the model performance
print('Mean Accuracy: get_ipython().run_line_magic(".3f", " std:(%.3f)' % (np.mean(n_scores), np.std(n_scores)))")


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, shares, random_state=0)
tree = DecisionTreeClassifier(random_state=0)
params = {
    'criterion':  ['gini', 'entropy'],
    'max_depth':  [None, 2, 4, 6, 8, 10],
    'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    'splitter': ['best', 'random']
}

model = GridSearchCV(
    estimator=tree,
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1,
)
model.fit(X_train, y_train)
print(model.best_params_)


model.score(X_test, y_test)
