import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pickle
stroke_df = pd.read_csv('data/stroke-data.csv')
print(stroke_df.head())
print(stroke_df.isnull().any())
print(stroke_df['bmi'].mean())
stroke_df['bmi'].fillna(stroke_df['bmi'].mean(), inplace=True)
print(stroke_df.isnull().any())
print(stroke_df.work_type.unique())
print(stroke_df.head())
stroke_df.drop(['id'], axis=1, inplace=True)
plt.figure(figsize=(8,6))
sns.heatmap(stroke_df.corr(), annot=True,
           vmax=1.0, linecolor='white',
            annot_kws={'fontsize':8 })
plt.show()
sns.countplot(y='gender', hue='stroke', data=stroke_df)
plt.show()
    
fig, ax = plt.subplots()
sns.kdeplot(stroke_df[stroke_df['stroke']==1]["age"], shade=True, color="red", label="Suffered", ax=ax)
sns.kdeplot(stroke_df[stroke_df['stroke']==0]["age"], shade=True, color="green", label="Not Suffered", ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("Density")
fig.suptitle("Age vs. Stroke possibility");
sns.countplot(y='Residence_type', hue='stroke', data=stroke_df)
plt.show()
sns.boxplot(x='hypertension', y='age', hue='stroke', data=stroke_df)
plt.show()
sns.boxplot(x='heart_disease', y='age', hue='stroke', data=stroke_df)
plt.show()
sns.countplot(x='ever_married', hue='stroke', data=stroke_df)
plt.show()
cat_cols = ['gender', 'ever_married','work_type','Residence_type','smoking_status']
stroke_df[cat_cols].nunique()
df_cat_encoded = pd.get_dummies(stroke_df[cat_cols], drop_first=True)
print(df_cat_encoded)
numeric_fields = list(set(stroke_df.columns) - set(cat_cols))
print(numeric_fields)
encoded_stroke_df = pd.concat([stroke_df[numeric_fields],df_cat_encoded], axis = 1)
print(cat_cols)
print(encoded_stroke_df.head())
x_fields = encoded_stroke_df.columns.tolist()
(x_fields.remove('stroke'))
print(x_fields)
X = encoded_stroke_df[x_fields]
Y = encoded_stroke_df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=102)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy_score(y_pred, y_test)
confusion_matrix(y_pred, y_test)
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)
pickle.dump(log_reg, open('models/stroke_predict_lg.pkl', 'wb'))
x1 = X_test[:1]
print(y_test.shape)
print(y_pred.sum())
print(x1.to_dict())
print(x1.shape)
clf = DecisionTreeClassifier(class_weight='balanced').fit(X_train, y_train)
y_hat = clf.predict(X_test)
print(y_hat.sum())
accuracy_score(y_hat, y_test)
print(classification_report(y_test, y_hat))
roc_auc_score(y_test, y_hat)
pickle.dump(clf, open('models/stroke_predict_dtc.pkl', 'wb'))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=202)
rfc.fit(X, Y)
y_hat_rfc = rfc.predict(X_test)
accuracy_score(y_test, y_hat_rfc)
roc_auc_score(y_hat_rfc, y_test)
print(classification_report(y_test, y_hat_rfc))
pickle.dump(rfc, open('models/stroke_predict_rfc.pkl', 'wb'))
y_hat_rfc.sum()
y_hat_rfc.size