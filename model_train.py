

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



############################################################################################
############ Feature Scaling

# create x,y
x = Df[cancer.feature_names]
y = Df['class']


from sklearn.preprocessing import StandardScaler

# standardisation
scaler = StandardScaler()
# apply scaler
x = scaler.fit_transform(x)


############################################################################################
############ split data set

#startified
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

x_train.shape, y_train.shape
x_test.shape, y_test.shape


# check for stratify
y_train.value_counts()
y_test.value_counts() 


############################################################################################
############ svm training 

# models import
from sklearn import svm

# classifier
classifier = svm.LinearSVC()

# fit
classifier.fit(x_train, y_train,)

# predict
y_pred = classifier.predict(x_test)


# classifiacation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=cancer.target_names)
disp.plot()


############################################################################################
############ LR training 

# models import
from sklearn.linear_model import LogisticRegression

# classifier
classifier = LogisticRegression()

# fit
classifier.fit(x_train, y_train)

# predict
y_pred = classifier.predict(x_test)


# classifiacation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=cancer.target_names)
disp.plot()


############################################################################################
############ manual optimization

# models import
from sklearn.linear_model import LogisticRegression

# L1
classifier = LogisticRegression(penalty='l1',
                                 solver='liblinear',
                                  C=3,
                                   max_iter=100,
                                    tol=0.01 )

# L2
classifier = LogisticRegression(penalty='l2',
                                 solver='lbfgs',
                                  C=4,
                                   max_iter=1000,
                                    tol=0.001 )

# fit
classifier.fit(x_train, y_train)

# predict
y_pred = classifier.predict(x_test)


# classifiacation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
