!pip install lazypredict --user
import lazypredict

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
pd.set_option('display.max_columns', 500)

df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()

df['TotalCharges'].describe()
df[df['TotalCharges']==' '].head()
df['TotalCharges'] = df['TotalCharges'].str.strip().replace('', 0).astype('float').copy()
df['TotalCharges'].dtype
target = df['Churn'].copy()
target_label = 'Churn'
cats = [col for col in df.columns if (df[col].dtype == 'object') & (col not in ['customerID'])]
nums = [col for col in df.columns if df[col].dtype != 'object']
print(cats)
print(nums)

df[cats].describe()


for col in cats:
    print(col)
    sns.countplot(data=df, x=col)
    plt.title(col)
    plt.tight_layout()
    plt.show()
    
    
df['tenureYear'] = df['tenure']/12
df['AnnualCharges'] = df['MonthlyCharges']*12
coef = np.polyfit(df['tenureYear'], df['AnnualCharges'], 1)
df['residual'] = df['AnnualCharges'] - (coef[0]*df['tenureYear'] + coef[1])

columns = [col for col in df.columns if col not in ['customerID', target_label]]
cats = [col for col in columns if df[col].dtype == 'object']
nums = [col for col in columns if df[col].dtype != 'object']
X = df[columns].copy()
y = df[target_label].copy()
X.shape, y.shape

mapping_churn = {
    'No' : 0,
    'Yes' : 1
}

y = y.map(mapping_churn)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X.copy(), y.copy(), test_size=0.25, random_state=42, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Z-score
# print(f'Rows before filtering outliers: {len(X_train)}')
# filtered_entries = np.array([True] * len(X_train))
# for col in X_train.columns:
#     zscore = abs(st.zscore(X_train[col]))
#     filtered_entries = (zscore < 3) & filtered_entries # keeping absolute z score under 3
# X_train = X_train[filtered_entries] # filter z score under 3
# print(f'Rows after filtering outliers: {len(X_train)}')

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()scaler.fit(X_train[nums])
X_train_scaled = scaler.transform(X_train[nums])
X_train[nums] = X_train_scaled
X_train.head()

from sklearn.preprocessing import LabelEncoder

labels = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
encoders = []
for i,label in enumerate(labels):
    i = LabelEncoder()
    i.fit(X_train[label])
    X_train[label] = i.transform(X_train[label])
    encoders.append(i)
    
from sklearn.preprocessing import OneHotEncoder
onehotcats = [col for col in cats if col not in labels]
ohe = OneHotEncoder(drop='first', sparse=False).fit(X_train[onehotcats])
ohe.get_feature_names(onehotcats)

X_train_ohe = ohe.transform(X_train[onehotcats])
to_merge = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names(onehotcats))
X_train = X_train.reset_index().drop('index', axis=1)
X_train[ohe.get_feature_names(onehotcats)] = to_merge
X_train = X_train[[col for col in X_train if col not in onehotcats]].copy()
X_train.head()

pd.Series(y_train).value_counts()
from imblearn import over_sampling
X_over_SMOTE, y_over_SMOTE = over_sampling.SMOTE(sampling_strategy=0.5, random_state=42).fit_resample(X_train, y_train)
print('SMOTE')
print(pd.Series(y_over_SMOTE).value_counts())

X_train, y_train = X_over_SMOTE, y_over_SMOTE
X_train.shape, y_train.shape

X_test_scaled = scaler.transform(X_test[nums])
X_test[nums] = X_test_scaled
X_test.head()labels = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for i, label in enumerate(labels):
    X_test[label] = encoders[i].transform(X_test[label])
X_test.head()

X_test_ohe = ohe.transform(X_test[onehotcats])
to_merge = pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names(onehotcats))
X_test = X_test.reset_index().drop('index', axis=1)
X_test[ohe.get_feature_names(onehotcats)] = to_merge
X_test = X_test[[col for col in X_test if col not in onehotcats]].copy()
X_test.head()


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
rf.fit(X_train, y_train)

brt = BorutaPy(rf, n_estimators='auto', random_state=42)
brt.fit(np.array(X_train), np.array(y_train))
brt_ranking = brt.ranking_
plt.figure(figsize=(8,8))
sns.scatterplot(y=[col for col in X_train.columns.values], x=brt_ranking, hue=brt_ranking)

selected_features = {}
for i, col in enumerate(X_train.columns):
    if brt_ranking[i] <= 2:
        selected_features[col] = brt_ranking[i]
selected_features

features = [k for k in selected_features.keys()]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
features_cat = [col for col in np.concatenate((labels,ohe.get_feature_names(onehotcats))) if col in features]
select = SelectKBest(score_func=chi2, k=5)
selector = select.fit(X_train[features_cat], y_train)scores = pd.DataFrame(features_cat)
scores['score'] = selector.scores_
scores = scores.sort_values('score', ascending=False)
sns.barplot(data=scores, x='score', y=0)
plt.title('Chi-Squared Test Statistic')
plt.ylabel('')
plt.tight_layout()
plt.show()


filtered_score = scores[scores['score']<=200]
filtered_score[0].valuesfeatures1 = [col for col in features if col not in filtered_score[0].values]
features1

temp = X_train[features1].copy()
temp[target_label] = y_train.copy()
corr = temp.corr(method='pearson')#.sort_values(y_train_label, ascending=False)
plt.figure(figsize=(14,14))
ax = sns.heatmap(corr, annot=True, fmt='.2f', vmin=-1, vmax=1, cmap='coolwarm')
plt.tight_layout()
plt.show()to_drop = [
    'OnlineSecurity_No internet service',
    'OnlineBackup_No internet service',
    'DeviceProtection_No internet service',
    'TechSupport_No internet service',
    'StreamingTV_No internet service',
    'StreamingMovies_No internet service',
    'tenure',
    'TotalCharges',
    'MonthlyCharges',
]

features2 = [f for f in features1 if f not in to_drop]
temp = X_train[features2].copy()
temp[target_label] = y_train.copy()
corr = temp.corr(method='pearson')#.sort_values(y_train_label, ascending=False)

plt.figure(figsize=(12,12))
ax = sns.heatmap(corr, annot=True, fmt='.2f', vmin=-1, vmax=1, cmap='coolwarm')
plt.tight_layout()
plt.show()

to_drop = [
    'InternetService_Fiber optic',
    'InternetService_No',
    'AnnualCharges'
]

features3 = [f for f in features2 if f not in to_drop]
temp = X_train[features3].copy()
temp[target_label] = y_train.copy()
corr = temp.corr(method='pearson')#.sort_values(y_train_label, ascending=False)

plt.figure(figsize=(10,10))
ax = sns.heatmap(corr, annot=True, fmt='.2f', vmin=-1, vmax=1, cmap='coolwarm')
plt.tight_layout()
plt.show()

from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import recall_score

clf = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=recall_score)
models, predictions = clf.fit(X_train[feat], X_test[feat], y_train, y_test)
print(models)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score, roc_auc_score


from sklearn.utils.extmath import softmax
from sklearn.metrics.pairwise import pairwise_distances

def predict_proba(self, X):
    distances = pairwise_distances(X, self.centroids_, metric=self.metric)
    probs = softmax(distances)
    return probs
from sklearn.model_selection import cross_val_score, KFold
def model_evaluation(model, X_train, y_train, scoring='recall', cv=5):
    cv_results = cross_val_score(model, X_train[feat], y_train, scoring=scoring, cv=cv)
    avg_res = abs(np.mean(cv_results))
    return avg_res
    
    qda=QuadraticDiscriminantAnalysis()
gnb=GaussianNB()
nc=NearestCentroid()
sgd=SGDClassifier()
pac=PassiveAggressiveClassifier()

models=[
    qda,
    gnb,
    nc,
    sgd,
    pac,
]
results = []
print('RECALL SCORES ON TRAINING DATA:\n')
for i, model in enumerate(models):
    result = model_evaluation(model, X_train[feat], y_train)
    print(model,':',result)
    
print('ROC AUC SCORES ON TRAINING DATA:\n')
for i, model in enumerate(models):
    if model!=nc:
        result = model_evaluation(model, X_train[feat], y_train, scoring='roc_auc')
    else:
        kf = KFold(n_splits=5, random_state=None, shuffle=False)
        res = []
        for train_index, test_index in kf.split(X_train, y=y_train):
            X1, X2 = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
            y1, y2 = y_train[train_index], y_train[test_index]
            model.fit(X1, y1)
            res.append(roc_auc_score(y2, predict_proba(nc, X2)[:,0]))
        result = np.mean(res)
    print(model,':',result)

test_results = []
print('RECALL SCORES ON TEST DATA:\n')
for model in models:
    model.fit(X_train[feat], y_train)
    y_pred = model.predict(X_test[feat])
    test_results.append(recall_score(y_test, y_pred))
for i, model in enumerate(models):
    print(model, ':', test_results[i])
    
test_results = []
print('ROC AUC SCORES ON TEST DATA:\n')
for model in models:
    model.fit(X_train[feat], y_train)
    y_pred = model.predict(X_test[feat])
    test_results.append(roc_auc_score(y_test, y_pred))
for i, model in enumerate(models):
    print(model, ':', test_results[i])

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
var_smoothing = np.logspace(0,-9, num=1500)
hyperparameters = dict(var_smoothing=var_smoothing)
gs = GridSearchCV(estimator=gnb, param_grid=hyperparameters, verbose=1, cv=5, n_jobs=-1, scoring='recall')
model1 = gs.fit(X_train[feat], y_train)
y_pred1 = model1.predict(X_test[feat])
print('Recall Score  :', recall_score(y_test, y_pred1))
print('ROC AUC Score :', roc_auc_score(y_test, y_pred1))
cm = confusion_matrix(y_test, y_pred1, labels=sorted(model1.classes_, reverse=True))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(model1.classes_, reverse=True))
disp.plot(cmap='Blues')
plt.tight_layout()
plt.show()

reg_param = np.linspace(0,0.5, num=1500)
hyperparameters = dict(reg_param=reg_param)
gs = GridSearchCV(estimator=qda, param_grid=hyperparameters, verbose=1, cv=5, n_jobs=-1, scoring='recall')
model2 = gs.fit(X_train[feat], y_train)

y_pred2 = model2.predict(X_test[feat])
print('Recall Score  :', recall_score(y_test, y_pred2))
print('ROC AUC Score :', roc_auc_score(y_test, y_pred2))

print(classification_report(y_test, y_pred2, labels=sorted(model2.classes_, reverse=True)))

cm = confusion_matrix(y_test, y_pred2, labels=sorted(model2.classes_, reverse=True))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(model2.classes_, reverse=True))
disp.plot(cmap='Blues')
plt.tight_layout()
plt.show()
