#!/usr/bin/env python
# coding: utf-8

# Source Dataset : https://www.kaggle.com/ajay1735/hmeq-data

# # Problem Statement

# **Context** <br>
# Departemen kredit konsumen sebuah bank ingin mengotomatisasi proses pengambilan keputusan untuk persetujuan jalur kredit ekuitas rumah. Untuk melakukan ini, mereka akan mengikuti rekomendasi dari Equal Credit Opportunity Act untuk membuat model penilaian kredit yang diturunkan secara empiris dan sehat secara statistik. Model ini akan didasarkan pada data yang dikumpulkan dari pemohon yang baru saja diberikan kredit melalui proses penjaminan pinjaman saat ini. Model akan dibangun dari alat pemodelan prediktif, tetapi model yang dibuat harus cukup dapat diinterpretasikan untuk memberikan alasan untuk setiap tindakan yang merugikan (penolakan).
# 
# **Content** <br>
# Dataset Ekuitas Rumah (HMEQ) berisi informasi dasar dan kinerja pinjaman untuk 5.960 pinjaman ekuitas rumah baru-baru ini. Target (BAD) adalah variabel biner yang menunjukkan apakah pelamar akhirnya gagal atau benar-benar tunggakan. Hasil buruk ini terjadi pada 1.189 kasus (20%). Untuk setiap pelamar, 12 variabel input dicatat.
# 
# **Goals** <br>
# Memprediksi apakah seseorang akan gagal membayar pinjaman atau tidak, berdasarkan atribut yang diberikan.

# ## Attribute Overview 
# - BAD: 1 = pemohon gagal dalam pinjaman atau tunggakan yang serius; 0= pinjaman yang dibayar pemohon (Variabel Target){Nominal Biner Assimetris}
# - LOAN: Jumlah permintaan pinjaman{Rasio-Skala Numerik}
# - MORTDUE: Jumlah yang harus dibayar pada hipotek yang ada{Rasio-Skala Numerik}
# - VALUE: Nilai properti saat ini{Rasio-Skala Numerik}
# - REASON: DebtCon = konsolidasi hutang; homelmp = Perbaikan rumah{Nominal}
# - JOB: Kategori pekerjaan{Nominal}
# - YOJ: Tahun di pekerjaan sekarang
# - DEROG: Jumlah laporan penghinaan utama
# - DELINQ: Jumlah kredit yang menunggak
# - CLAGE: Usia batas kredit tertua dalam beberapa bulan
# - NINQ: Jumlah pertanyaan kredit terbaru
# - CLNO: Jumlah jalur kredit
# - DEBTINC: Rasio utang terhadap pendapatan{Rasio-Skala Numerik}

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('load_ext', 'autotime')


# In[2]:


# Pyplot and seaborn style
sns.set(style="whitegrid")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
colormap = sns.diverging_palette(220, 10, as_cmap=True)


# # Data Exploration 

# Pada tahap ini dilakukan eksplorasi data, seperti dimensi data, tipe data setiap kolom, statistik deskriptif data, mengecek apakah terdapat missing value, dan duplikat data. 

# ## Load  Dataset

# In[3]:


# Load dataset 
df = pd.read_csv('hmeq.csv')
df.head()


# ## Data Information 

# In[4]:


# dataframe information
df.info()


# ## Statistics Description

# In[5]:


# statistics description 
df.describe()


# In[6]:


# statistics for categorical data
df.select_dtypes('object').describe()


# ## Check Null and Misssing Values

# Periksa missing values untuk setiap featurem hitung jumlah dan persentasenya.

# In[7]:


# number of missing data
null_values = df.isna().sum()

# Round the percentage result
pct_null_values = round((df.isna().sum()/df.shape[0]*100), 2)

# Create a dataframe for the total amount and percentage of missing data
df_null_values = pd.DataFrame(data=null_values, columns=['Total']).reset_index()
df_null_values.rename(columns={'index':'Feature'}, inplace=True)

# Assign pertencage to dataframe
df_null_values['Percentage'] = pct_null_values.values

# filter for missing data more than zero
df_null_values = df_null_values[df_null_values['Total'] > 0]
df_null_values


# In[8]:


# Plotting number of missing data
fig, ax = plt.subplots(figsize=(15,6))

g = sns.barplot(x = 'Feature',y='Percentage',data=df_null_values,ax=ax, 
               palette=sns.color_palette("Blues_d", n_colors=13, desat=1))

x = np.arange(len(df_null_values['Feature']))
y = df_null_values['Percentage']

for i, v in enumerate(y):
    ax.text(x[i]-0.3, v+2, str(v)+'%', fontsize = 12, color='gray', fontweight='bold')
    

text = '''
There are 11 (84.61%) features that have missing value.
9 missing value in numeric data,
2 missing value in non-numeric data(text or categorical)

Feature with more than 10% missing value is  
DEBTINC (21.26%) and DEROG (11.88%)
'''
ax.text(0,60,text,horizontalalignment='left',color='gray',fontsize=14,fontweight='normal')
ax.set_title('Missing values distribution', color='black', fontsize=20, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
ax.set_ylim(0,100)
plt.show()


# ## Check Duplicated Data

# Cek apakah terdapat data yang duplikat, jika ada maka hapus data tersebut.

# In[9]:


# Check duplicated data
df[df.duplicated()]


# Terlihat bahwa data yang kita miliki tidak mempunyai data duplikat.

# # Data Cleansing

# Pada tahap ini lakukan pembersihan data, mulai dari handling missing value, inconsistent data, outliers, dan konversi  tipe data setiap feature sesuai dengan seharusnya.

# ## Check Cardinality

# Cek apakah terdapat data yang tidak konsisten pada setiap feature. Periksa juga jumlah unik data pada setiap feature.

# In[10]:


# get all categorical columns
col_cat = df.select_dtypes('object').columns

# loops for categorical data
for i in col_cat:
    print(df[i].value_counts())
    print()


# In[11]:


# The number of unique values of each feature
for i in df.columns:
    print(i, ':', df[i].nunique())


# ## Handling Missing Values

# In[12]:


# get all numerical columns
num_col = df.select_dtypes('number').columns

plt.figure(figsize=(15, 20))
pos = 1

# Loops for numerical feature 
for i in num_col:
    plt.subplot(5, 3, pos)
    sns.boxplot(data=df, x=i)
    pos += 1


# In[13]:


plt.figure(figsize=(18, 20))
pos = 1

# Loops for numerical feature 
for i in num_col:
    plt.subplot(5, 3, pos)
    sns.histplot(data=df, x=i)
    pos += 1


# Karena terdapat missing value, maka kita harus mengisi nilai tersebut. Pada data kategorikal kita bisa mengisinya menggunakan modus. Terdapat beberapa skenario untuk melakukan hal tersebut.
# - REASON : Feature ini merupakan alasan pemohon mengajukan pinjaman. Ada dua alasan, yaitu konsolidasi hutang dan perbaikan rumah. Karena terdapat data yang hilang, bisa diasumsikan bahwa pemohon tidak menyebutkan alasan dirinya meminjam. Jadi pada feature ini bisa diisii alasan lain (Other)
# - JOB : feature ini merupakan pekerjaan pemohon. Terdapat enam nilai unik pada feeature ini, dan "Other" yang paling banyak muncul. Maka kita bisa mengisi dengan nilai ini.    
# - Karena data yang kita miliki terdapat banyak outliers, maka kita bisa gunakan nilai median untuk mengisi missing values (data numerik).  

# In[14]:


# Replace REASON with Other
df['REASON'].fillna('Other', inplace=True)
# FIlling JOB with mode 
df['JOB'].fillna(df['JOB'].mode()[0], inplace=True)


# In[15]:


# Filling missing values with median
df.fillna(df.median(), inplace=True)


# In[16]:


# Make sure there are no missing values
df.isna().any()


# ## Handling Outliers

# Karena terdapat banyak outliers, maka kita akan menghapus menggunakan metode IQR (Inter-quartile Range).

# In[17]:


# Define function to remove outliers
def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_final


# In[18]:


# Call function to remove outliers
df_drop_outliers = remove_outlier_IQR(df)

# Number of outliers has been removed
number_drop_outliers = df.shape[0]-df_drop_outliers.shape[0]
# percentage number of outliers
pct_number_drop_outliers = (number_drop_outliers/df.shape[0])
print("Jumlah data yang dihapus ada {} baris ({:.2%}) dari keseluruhan data.".format(number_drop_outliers,pct_number_drop_outliers))


# Karena data outliers yang dihapus menyebabkan kita kehilangan banyak data (lebih dari 50%), maka kita tidak akan menghapus data outliers. Kita akan menangani data tersebut pada tahap Feature Engineering. 

# ## Data Type Transformation

# ### DELINQ, NINQ, CLNO

# In[19]:


# Create list for DELINQ, NINQ, CLNO feature
col = ['DELINQ', 'NINQ', 'CLNO']

# Loops for col list
for i in col:
    print(df[i].unique())


# Karena pada feature DELINQ, NINQ, dan CLNO itu seharunya angka integer, maka kita ubah tipe datanya menjadi integer.

# In[20]:


# Convert DELINQ, NINQ, CLNO to integer
for i in col:
    df[i] = df[i].astype('int')


# ### REASON AND JOB

# In[21]:


# Get all categorical feature
col_cat = df.select_dtypes('object').columns

# Convert categorical feature to category data type
for i in col_cat:
    df[i] = df[i].astype('category')


# Feature JOB dan REASON juga kita ubah menjadi tipe data category karena data tersebut termasuk data kategorikal.  

# In[22]:


# Make sure that data type has changed
df.dtypes


# # Exploratory Data Analysis

# Pada tahap EDA ini dilakukan eksplorasi pada data, seperti distribusi pemohon pinjaman, distribusi jumlah pinjaman dan lain-lain.  

# ## BAD

# In[23]:


# Plotting distribution of applicant defaulted on loan or paid loan
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='BAD')
plt.title('Distribution of Applicant Defaulted on Loan or Paid Loan', fontsize=20, color='black', pad=15)
plt.show()


# ## LOAN

# In[24]:


# Plotting amount of the loan request distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='LOAN', kde=True, hue='BAD')
plt.title('Amount of the Loan Request Distribution', fontsize=20, color='black', pad=15)
plt.tight_layout()
plt.show()


# ## MORTDUE

# In[25]:


# Plotting Distribution of MORTDUE by Loan Applicant
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='MORTDUE', kde=True, hue='BAD')
plt.title('Distribution of MORTDUE by Loan Applicant', fontsize=20, color='black', pad=15)
plt.tight_layout()
plt.show()


# ## VALUE

# In[26]:


# Plotting Distribution of VALUE by Loan Applicant
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='VALUE', kde=True, hue='BAD')
plt.title('Distribution of VALUE by Loan Applicant', fontsize=20, color='black', pad=15)
plt.tight_layout()
plt.show()


# ## REASON

# In[27]:


# Plotting Distribution of Loan Applicant Reasons
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='REASON', hue='BAD')
plt.title('Distribution of Loan Applicant Reasons', fontsize=20, color='black', pad=15)
plt.tight_layout()
plt.show()


# ## JOB

# In[28]:


# Plotting Distribution of JOB
plt.figure(figsize=(10, 6))
df['JOB'].value_counts().sort_values(ascending=False).plot(kind='bar')
plt.title('Distribution of JOB', fontsize=20, color='black', pad=15)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ## DEROG

# In[29]:


# Plotting DEROG feature
sns.jointplot(x="DEROG", y="BAD", data=df, kind="reg")
plt.show()


# ## DELINQ

# In[30]:


# Plotting DELINQ feature
sns.jointplot(x="DELINQ", y="BAD", data=df, kind="reg")
plt.show()


# ## CLAGE

# In[31]:


# Plotting Distribution of CLAGE by Loan Applicant
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='CLAGE', kde=True, hue='BAD')
plt.title('Distribution of CLAGE by Loan Applicant', fontsize=20, color='black', pad=15)
plt.tight_layout()
plt.show()


# ## NINQ

# In[32]:


# Plotting Distribution of NINQ
plt.figure(figsize=(10, 6))
df['NINQ'].value_counts().sort_values(ascending=False).plot(kind='bar')
plt.title('Distribution of NINQ', fontsize=20, color='black', pad=15)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ## CLNO

# In[33]:


# Plotting CLNO feature
sns.jointplot(x="CLNO", y="BAD", data=df, kind="reg")
plt.show()


# ## DEBTINC

# In[34]:


# Plotting DEBTINC feature
sns.jointplot(x="DEBTINC", y="BAD", data=df, kind="reg")
plt.show()


# ## Correlation Features

# In[35]:


# Correlation heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[36]:


# Correlation feature using pairplot
sns.pairplot(df)
plt.show()


# # Feature Engineering

# Pada tahap feature engineering ini akan dilakukan pembuatan feature baru dari feature yang sudah ada, merubah data menjadi format yang dapat dibaca oleh model ML, dan rescaling.

# In[37]:


# Add new feature from existing data
df['PROBINC'] = df['MORTDUE']/df['DEBTINC'] # adding new feature, (current debt on mortgage)/(debt to income ratio)


# ## Feature Encoding

# ### One Hot Encoding

# Lakukan one hot encoding pada feature non-numeric, yaitu feature JOB dan REASON dan gabungkan hasil encoding dengan dataframe awal agar lebih mudah dipahami.

# In[38]:


# One Hot Encoding using dummies
encoded = pd.get_dummies(df[['REASON', 'JOB']], prefix=['reason', 'job'])

# Merge dataframe
df = pd.concat([df, encoded], axis=1)

# Delete REASON and JOB features
df.drop(['REASON', 'JOB'], axis=1, inplace=True)
df.head()


# ## Feature Scaling

# Pisahkan antara feature dan target.

# In[39]:


# Separate features and targets
x = df.drop(['BAD'], axis=1)
y = df['BAD']


# Karena pada data yang kita miliki terdapat banyak outliers, maka kita lakukan scaling menggunakan RobustScaling.

# In[40]:


# Scaling features
transformed = RobustScaler().fit_transform(x)

# Create a scaled dataframe
df_scaled = pd.DataFrame(transformed, columns=x.columns)
df_scaled['BAD'] = y
df_scaled


# # Sampling Dataset

# ## Separating Train and Test Set 

# Pada tahap ini kita akan memisalkan data latih dan data uji untuk nantinya digunakan pada model yang akan kita latih.

# In[41]:


# Separate features and target

X = df_scaled.drop('BAD', axis=1)
y = df_scaled['BAD']


# In[42]:


# Separate train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[43]:


# Dimension train and test set
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# ## Imbalanced Class

# Karena target yang kita punya class nya tidak terdistribusi normal, maka kita akan lakukan over sampling dengan menggunakan metode SMOTE. 

# In[44]:


# Check target distribution 
pd.Series(y_train).value_counts()


# ### Over Sampling Using SMOTE

# In[45]:


# Applying SMOTE
X_over, y_over = SMOTE().fit_resample(X_train, y_train)

# Make sure classes are evenly distributed
pd.Series(y_over).value_counts()


# # Modelling

# Pada tahap modeling ini kita melakukan beberapa tahapan, yaitu memilih model yang terbaik, melakukan tuning hyperparamater pada model yang dipilih, lakukan evaluasi model, dan simpan model.

# ## Choose The Best Classifier

# Pada tahap ini kita akan menggunakan beberapa metode yang nantinya kita lihat performanya, model dengan performa terbaik akan kita lakukan tuning pada tahap selanjutnya.

# In[46]:


# Define classier to a list
classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(), 
    AdaBoostClassifier(), 
    ExtraTreesClassifier(), 
    GradientBoostingClassifier(), 
    BaggingClassifier(),
    KNeighborsClassifier(), 
    LogisticRegression(),
    GaussianNB(), 
    SVC()]


# In[47]:


# Create dataframe for model performance result
model_performance = pd.DataFrame(columns=['Method', 'accuracy', 'precision', 'recall', 'f1-score'])


# In[48]:


# Train data for each model
for classifier in classifiers:
    classifier.fit(X_over, y_over)
    y_pred = classifier.predict(X_test)

    method = str(type(classifier)).split('.')[-1][:-2]

    # accuracy score
    acc = accuracy_score(y_test, y_pred)

    # precision score
    pre = precision_score(y_test, y_pred)

    # recall score
    re = recall_score(y_test, y_pred)

    # f1 score
    f1 = f1_score(y_test, y_pred)

    # Append model performance metrics to dataframe
    model_performance = model_performance.append({
        'Method':method,
        'accuracy':acc,
        'precision':pre,
        'recall':re,
        'f1-score':f1}, ignore_index=True)


# In[49]:


# Sort accuracy, precision, recall, and f2-score in descending order
model_performance.sort_values(['accuracy', 'precision', 'recall', 'f1-score'], ascending=False).reset_index(drop=True)


# Pada hasil percobaan yang sudah dilakukan, terlihat bahwa model ExtraTreesClassifier memiliki performa terbaik, maka kita akan lakukan tuning pada model ini.

# ## Tuning Hyperparameters

# Lakukan tuning hyperparameters pada model ExtraTreesClassifier untuk menghasilkan performa performa terbaik.

# In[50]:


clf = ExtraTreesClassifier()

# Define parameters
params = {
    'n_estimators': range(0,201,25),
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap':[True, False],
    }

# Applying grid search
grid_result = GridSearchCV(clf, params, scoring='f1', cv=5)


# In[51]:


# Fit grid search 
grid_result.fit(X_over, y_over)

# Display best score and best parameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# ## Model Performance

# Setelah didapatkan parameter terbaik, maka kita tinggal melatih model yang sudah kita pilih menggunakan parameter tersebut.

# In[52]:


# Applying best parameters to model
clf = ExtraTreesClassifier(**grid_result.best_params_)
clf.fit(X_over, y_over)


# ### Confusion Matrix

# In[53]:


# Prediction data test
y_pred = clf.predict(X_test)

# Confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
            columns = ['pred 0', 'pred 1'],
            index = ['act 0', 'act 1'])
cm_df


# Pada hasil confusion matrix didapatkan hasil 1368 terprediksi benar class 0 (pemohon gagal membayar) dan 312 terprediksi class 1 (pemohon membayar).

# ### Classification Report

# In[54]:


# Classification report model
print(classification_report(y_test, y_pred))


# Pada hasil diatas, baik accuracy, precision, recall dan f1-score memiliki nilai yang cukup baik, yaitu diatas 75%, dengan nilai akurasi 94%.

# ### Feature Importance

# In[55]:


# Assign feature importance to the dataframe
feature_importance = pd.DataFrame(
    clf.feature_importances_, 
    index=X.columns, 
    columns=['importance']).sort_values('importance', ascending=False).reset_index()

# Plotting feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='importance', y='index', color='#03bafc')
plt.title('Feature Importance', fontsize=20, color='black', pad=15)
plt.show()


# ## Save Model

# In[56]:


# Save model with joblib
joblib.dump(clf,'extratree')


# # Summary

# ### Korelasi Feature dan Target
# - Pada data yang dilatih class yang dominan yaitu class 0 (pemohon berhasil membayar)
# - Distribusi jumlah pinjaman antara class 0 dan class 1 memiliki distribusi yang sama, maka bisa dikatakan jumlah pinjaman tidak berpengaruh kepada pemohon gagal membayar
# -  Distribusi pemohon yang gagal membayar dan tidak cenderung sama pada pada setiap alasan peminjaman, jadi bisa dikatakan alasan peminjaman tidak berpengaruh kepada gagalnya pembayaran pinjaman
# - Jumlah laporan penghinaan utama, jumlah kredit menunggak cukup tinggi memiliki korelasi positif yang cukup tinggi pada faktor kegagalan pemohon membayar. Karena secara umum semakin banyak kredit mengunggak maka semakin sulit juga untuk membayar.

# ### Model Performance
# - Pada hasil percobaan didapatkan bahwa model Extra Tree Classifier memiliki performa terbaik dari beberapa model yang digunakan, dengan akurasi mencapai 94%.
# - Dilakukan tuning hyperparameter untuk memilih parameter terbaik yang akan digunakan dalam training model.
# - Dari hasil confusion matrix didapatkan TP dan TN yang tinggi, maka bisa di katakan model dapat mengklasifikasi dengan cukup baik dengan akurasi mencapai 94%. 
