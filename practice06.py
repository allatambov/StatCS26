import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

### Подготовка данных ###

df = pd.read_csv("city24.csv")
df.head()

data = df[["Decibel_Level", "Green_Space_Area", "Air_Quality_Index",
           "Cost_of_Living_Index", "Healthcare_Index"]]

X = StandardScaler().fit_transform(data)
data_scaled = pd.DataFrame(X, columns = data.columns)
data_scaled.head()

R = data_scaled.corr().round(2)
Sigma = data_scaled.cov().round(2)

### Реализация МГК ###

p = data_scaled.shape[1]
pca_names = ["PC" + str(i) for i in range(1, p + 1)]

pca = PCA(n_components = p)
pca_res = pd.DataFrame(pca.fit_transform(data_scaled),
                       columns = pca_names)

print(pca_res.corr().round())

### Информативность и выбор числа компонент ###

pca_var = pca.explained_variance_
pca_var_ratio = pca.explained_variance_ratio_

print(pca_var.round(2))
print((pca_var_ratio * 100).round(2))
print(np.cumsum((pca_var_ratio * 100).round(2)))

plt.plot(range(1, p + 1), pca_var, "o-");
plt.title("Scree plot");
plt.xlabel("Number of components");
plt.ylabel("Eigenvalues (variances)");

### Матрица факторных нагрузок ###

P = pd.DataFrame(pca.components_).T
P.columns = pca_names
P.index = data.columns
print(P)

A = P * np.sqrt(pca_var)
print(A)

print((A ** 2).sum(axis = 0))
print((A ** 2).sum(axis = 1))
print("Для сравнения λ:", pca_var)

print(pd.DataFrame(np.dot(A, A.T).round(2)))
print("Covariance (unbiased covariance): \n", Sigma)

print((A ** 2).round(2))

i = A.loc["Decibel_Level", :]
j = A.loc["Air_Quality_Index", :]
print("Скалярное произведение:", np.dot(i, j).round(3))
print("Корреляция:", data_scaled.cov().loc["Decibel_Level",
                                  "Air_Quality_Index"].round(3))

### Работа в пространстве сниженной размерности ###

df["Index1"] = pca_res["PC1"]
df["Index2"] = pca_res["PC2"]
df[["Index1", "Index2"]].describe().round(2)

df.sort_values("Index1", ascending = False)

df[["Index1", "Index2", "Happiness_Score"]].corr().round(2)
df.plot.scatter(x = "Index1", y = "Index2");





