
# coding: utf-8

# # Create a logistic regression model to predict RIT1 mutation from gene expression data in TCGA

# In[1]:

import os
import urllib
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[2]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# ## Specify model configuration

# In[3]:

# We're going to be building a 'RIT1' classifier 
GENE = '6016'# RIT1


# *Here is some [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) regarding the classifier and hyperparameters*
# 
# *Here is some [information](https://ghr.nlm.nih.gov/gene/RIT1) about RIT1*

# ## Load Data

# In[4]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', 'download', 'expression-matrix.tsv.bz2')\nX = pd.read_table(path, index_col=0)")


# In[5]:

get_ipython().run_cell_magic('time', '', "path = os.path.join('..', 'download', 'mutation-matrix.tsv.bz2')\nY = pd.read_table(path, index_col=0)")


# In[6]:

y = Y[GENE]


# In[7]:

# The Series now holds RIT1 Mutation Status for each Sample
y.head(6)


# In[8]:

# Here are the percentage of tumors with RIT1
y.value_counts(True)


# ## Set aside 20% of the data for testing

# In[9]:

# Typically, this can only be done where the number of mutations is large enough
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
'Size: {:,} features, {:,} training samples, {:,} testing samples'.format(len(X.columns), len(X_train), len(X_test))


# ## Feature Standardization

# In[10]:

get_ipython().run_cell_magic('time', '', 'scale_pre = StandardScaler()\nX_train_scale = scale_pre.fit_transform(X_train)\nX_test_scale = scale_pre.transform(X_test)')


# ## Reducing noise via PCA

# In[11]:

get_ipython().run_cell_magic('time', '', 'n_components = 30\npca = PCA(n_components=n_components, random_state=0)\nX_train_pca = pca.fit_transform(X_train_scale)\nX_test_pca = pca.transform(X_test_scale)')


# In[12]:

get_ipython().run_cell_magic('time', '', 'scale_post = StandardScaler()\nX_train_scale = scale_post.fit_transform(X_train_pca)\nX_test_scale = scale_post.transform(X_test_pca)')


# ## Parameters

# In[13]:

param_grid = {
    'alpha': [10 ** x for x in range(-4, 1)],
    'l1_ratio': [0, 0.2, 0.5, 0.8, 1],
}


# In[14]:

get_ipython().run_cell_magic('time', '', "clf = SGDClassifier(random_state=0, class_weight='balanced', loss='log', penalty='elasticnet')\ncv = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')\ncv.fit(X = X_train_scale, y=y_train)")


# In[15]:

# Best Params
print('{:.3%}'.format(cv.best_score_))

# Best Params
cv.best_params_


# ## Visualize hyperparameters performance

# In[16]:

cv_result_df = pd.concat([
    pd.DataFrame(cv.cv_results_),
    pd.DataFrame.from_records(cv.cv_results_['params']),
], axis='columns')
cv_result_df.head(2)


# In[17]:

# Cross-validated performance heatmap
cv_score_mat = pd.pivot_table(cv_result_df, values='mean_test_score', index='l1_ratio', columns='alpha')
ax = sns.heatmap(cv_score_mat, annot=True, fmt='.1%')
ax.set_xlabel('Regularization strength multiplier (alpha)')
ax.set_ylabel('Elastic net mixing parameter (l1_ratio)');


# ## Use Optimal Hyperparameters to Output ROC Curve

# In[18]:

y_pred_train = cv.decision_function(X_train_scale)
y_pred_test = cv.decision_function(X_test_scale)

def get_threshold_metrics(y_true, y_pred):
    roc_columns = ['fpr', 'tpr', 'threshold']
    roc_items = zip(roc_columns, roc_curve(y_true, y_pred))
    roc_df = pd.DataFrame.from_items(roc_items)
    auroc = roc_auc_score(y_true, y_pred)
    return {'auroc': auroc, 'roc_df': roc_df}

metrics_train = get_threshold_metrics(y_train, y_pred_train)
metrics_test = get_threshold_metrics(y_test, y_pred_test)


# In[19]:

# Plot ROC
plt.figure()
for label, metrics in ('Training', metrics_train), ('Testing', metrics_test):
    roc_df = metrics['roc_df']
    plt.plot(roc_df.fpr, roc_df.tpr,
        label='{} (AUROC = {:.1%})'.format(label, metrics['auroc']))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Predicting RIT1 mutation from gene expression (ROC curves)')
plt.legend(loc='lower right');


# ## Investigate the predictions

# In[20]:

X_transformed = scale_post.transform(pca.transform(scale_pre.transform(X)))


# In[21]:

predict_df = pd.DataFrame.from_items([
    ('sample_id', X.index),
    ('testing', X.index.isin(X_test.index).astype(int)),
    ('status', y),
    ('decision_function', cv.decision_function(X_transformed)),
    ('probability', cv.predict_proba(X_transformed)[:, 1]),
])
predict_df['probability_str'] = predict_df['probability'].apply('{:.1%}'.format)


# In[22]:

# Top predictions amongst negatives (potential hidden responders)
predict_df.sort_values('decision_function', ascending=False).query("status == 0").head(10)


# In[23]:

# Ignore numpy warning caused by seaborn
warnings.filterwarnings('ignore', 'using a non-integer number instead of an integer')

ax = sns.distplot(predict_df.query("status == 0").decision_function, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").decision_function, hist=False, label='Positives')


# In[24]:

ax = sns.distplot(predict_df.query("status == 0").probability, hist=False, label='Negatives')
ax = sns.distplot(predict_df.query("status == 1").probability, hist=False, label='Positives')

