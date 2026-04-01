from sklearn.datasets import load_iris
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve

# #I don't know how to get the kaggle thing to work...
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("sadiajavedd/social-media-user-activity-dataset")

# print("Path to dataset files:", path)

main_df = pd.read_csv("instagram_usage_lifestyle.csv")

df_encoded = main_df.copy()
for col in main_df.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded.drop('urban_rural', axis=1).values
y = df_encoded['urban_rural'].values

# print(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Correlation Coefficient.
corr_matrix = df_encoded.corr()

# Data needs to be heavily processed before any algorith can be done.
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot= True, fmt= ".2f", cmap= "coolwarm", center= 0, square= True, linewidths= 0.5)
plt.title('Correlation Coefficient Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()