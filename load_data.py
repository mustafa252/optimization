
# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
# split
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics


# load dataset
cancer = datasets.load_breast_cancer()
# show columns
cancer.keys()
# data
Data = cancer.data
# features
features = cancer.feature_names
# targets
targets = cancer.target  #  {0:Malignant, 1:benign}
target_names = cancer.target_names

# shapes
Data.shape, features.shape, targets.shape

# dataset into Dframe
Df = pd.DataFrame(data=Data, columns=features)
Df.columns
# add targets to the dframe
Df['class'] = targets
############################################################################################
############ data anlaysis

# info
Df.info()
# check null values
Df.isnull().sum()

############################################################################################
############ plot the count of values  

# quality + size
plt.rcParams['figure.figsize'] = [9,3]
plt.rcParams['figure.dpi'] = 300
# plot the count
count_plot = sns.countplot(x = targets, label='count')
# bar values
count_plot.bar_label(count_plot.containers[0], label_type='center')
# x ticks label
plt.xticks(Df['class'].unique(), cancer.target_names)




############################################################################################
############ plot Behaivor of the data


# pairPlot
sns.pairplot(Df, hue='class',
              vars=['mean radius', 'mean texture', 
                    'mean area', 'mean perimeter', 'mean smoothness'])




