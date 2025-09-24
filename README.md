
<br>

**\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]**


<br><br>

# 8- [Data Mining]()  / [K-Means: Non-Hierarchical Clustering]()



<!-- ======================================= Start DEFAULT HEADER ===========================================  -->

<br><br>


[**Institution:**]() Pontifical Catholic University of São Paulo (PUC-SP)  
[**School:**]() Faculty of Interdisciplinary Studies  
[**Program:**]() Humanistic AI and Data Science
[**Semester:**]() 2nd Semester 2025  
Professor:  [***Professor Doctor in Mathematics Daniel Rodrigues da Silva***](https://www.linkedin.com/in/daniel-rodrigues-048654a5/)

<br><br>

#### <p align="center"> [![Sponsor Quantum Software Development](https://img.shields.io/badge/Sponsor-Quantum%20Software%20Development-brightgreen?logo=GitHub)](https://github.com/sponsors/Quantum-Software-Development)


<br><br>

<!--Confidentiality statement -->

#

<br><br><br>

> [!IMPORTANT]
> 
> ⚠️ Heads Up
>
> * Projects and deliverables may be made [publicly available]() whenever possible.
> * The course emphasizes [**practical, hands-on experience**]() with real datasets to simulate professional consulting scenarios in the fields of **Data Analysis and Data Mining** for partner organizations and institutions affiliated with the university.
> * All activities comply with the [**academic and ethical guidelines of PUC-SP**]().
> * Any content not authorized for public disclosure will remain [**confidential**]() and securely stored in [private repositories]().  
>


<br><br>

#

<!--END-->




<br><br><br><br>



<!-- PUC HEADER GIF
<p align="center">
  <img src="https://github.com/user-attachments/assets/0d6324da-9468-455e-b8d1-2cce8bb63b06" />
-->


<!-- video presentation -->


##### 🎶 Prelude Suite no.1 (J. S. Bach) - [Sound Design Remix]()

https://github.com/user-attachments/assets/4ccd316b-74a1-4bae-9bc7-1c705be80498

####  📺 For better resolution, watch the video on [YouTube.](https://youtu.be/_ytC6S4oDbM)


<br><br>


> [!TIP]
> 
>  This repository is a review of the Statistics course from the undergraduate program Humanities, AI and Data Science at PUC-SP.
>
> ### ☞ **Access Data Mining [Main Repository](https://github.com/Quantum-Software-Development/1-Main_DataMining_Repository)**
>
>


<!-- =======================================END DEFAULT HEADER ===========================================  -->


<br><br>

## Theory Overview

<br>


K-Means is a non-hierarchical clustering algorithm that partitions data into \(k\) clusters, minimizing variance within clusters.

<br>

### Advantages
- Simple and efficient
- Scales well with large datasets
- Fast convergence

<br>

### Disadvantages
- Requires predefined number of clusters \(k\)
- Sensitive to initial centroids, leading sometimes to local minima
- Assumes spherical clusters of similar sizes
- Sensitive to outliers

<br>

### Elbow Method

The Elbow Method plots the Within-Cluster Sum of Squares (WCSS) versus number of clusters \(k\). The optimal \(k\) is indicated by the 'elbow' point where adding another cluster does not significantly reduce WCSS.

<br><br>


### Clustering Algorithms Overview

- K-Means (partitional, non-hierarchical)
- Hierarchical Clustering (agglomerative and divisive)
- DBSCAN (density-based)
- K-Medoids and others

<br><br>

## Step-by-Step K-Means Implementation

<br>

### Libraries Import and Dataset Loading

<br><br>


```python
# Import necessary libraries

# Importar bibliotecas necessárias

import pandas as pd  \# for data handling / para manipulação de dados
import matplotlib.pyplot as plt  \# for plotting / para plotagem
import seaborn as sns  \# enhanced visualization / visualização aprimorada
from sklearn.cluster import KMeans  \# KMeans algorithm / algoritmo KMeans
from sklearn.preprocessing import MinMaxScaler  \# normalization / normalização

# Load dataset

# Carregar o dataset

df = pd.read_csv('clientes-shopping.csv')
print(df.head())  \# Show first rows / mostrar primeiras linhas

```

<br><br>

### Data Preprocessing and Normalization

```python
# Drop unnecessary columns: CustomerID, Gender, Age for clustering

# Remover colunas irrelevantes para clusterização

df_cluster = df.drop(['CustomerID', 'Gender', 'Age'], axis=1)

# Normalize the features with MinMaxScaler

# Normalizar características usando MinMaxScaler

scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df_cluster), columns=df_cluster.columns)

print(df_norm.head())  \# Preview normalized data / visualizar dados normalizados
```


<br><br>


### Scatter Plot of Raw Data 

```python
# Scatter plot of Annual Income vs Spending Score, colored by Gender

# Scatter plot de Renda Anual vs Score de Gastos com legenda por Gênero

sns.set_style('dark')
palette = sns.color_palette('turquoise', 3)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k\$)', y='Spending Score (1-100)', hue='Gender', palette=palette)
plt.title('Annual Income vs Spending Score by Gender')
plt.show()
```


<br><br>


### Elbow Method to Determine Optimal K

<br>

```python
# Elbow Method to find the ideal number of clusters k

# Método do Cotovelo para definir o número ideal de clusters k

wcss = []
for k in range(1, 11):
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(df_norm)
wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', color='turquoise')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()
```

<br><br>

### Running K-Means with $\(k=5\)$

<br>

```python
# Apply KMeans with k=5 and random seed=42

# Aplicar KMeans com k=5 e seed=42

kmeans_model = KMeans(n_clusters=5, random_state=42)
clusters = kmeans_model.fit_predict(df_norm)

# Add clusters to original dataframe

# Adicionar clusters ao dataframe original

df['Cluster'] = clusters

print(df[['Annual Income (k\$)', 'Spending Score (1-100)', 'Cluster']].head())
```

<br><br>



### Scatter Plot with Clusters (Dark Mode, Turquoise Palette)

<br>

```python
# Scatter plot with clusters in different colors (turquoise palette)

# Scatter plot com clusters em cores diferentes (palette turquesa)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k\$)', y='Spending Score (1-100)',
hue='Cluster', palette='turquoise', legend='full')
plt.title('KMeans Clusters (k=5)')
plt.show()

```

<br><br>


### Scatter Plot with Gender Legend and Cluster Styles

<br>

```python
# Scatter plot with Gender as the legend and points styled by cluster

# Scatter plot com gênero na legenda e estilo pelos clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k\$)', y='Spending Score (1-100)',
hue='Gender', style='Cluster', palette=palette)
plt.title('Annual Income vs Spending Score by Gender and Cluster')
plt.show()

```

<br><br>

### Cluster Statistics

<br>


<!--

```


# Show descriptive statistics per cluster group

# Mostrar estatísticas descritivas por cluster

cluster_stats = df.drop(['CustomerID', 'Age'], axis=1).groupby('Cluster').describe()
print(cluster_stats)

```

---

### Display Centroid Coordinates

```


# Show coordinates of cluster centroids (in normalized scale)

# Mostrar coordenadas dos centróides dos clusters (normalizado)

print(kmeans_model.cluster_centers_)

```

---

### Silhouette Score for Cluster Validation

```

from sklearn.metrics import silhouette_score

# Silhouette score calculation for k=5

# Cálculo da silhueta para k=5

sil_score = silhouette_score(df_norm, clusters)
print(f'Silhouette Score for k=5: {sil_score:.3f}')

```

---

### Silhouette Scores for k=2 to k=10

```


# Silhouette scores for k=2 to k=10 to validate cluster choice

# Calcular pontuação da silhueta para k=2 a k=10

for n_clusters in range(2, 11):
model = KMeans(n_clusters=n_clusters, random_state=42)
preds = model.fit_predict(df_norm)
score = silhouette_score(df_norm, preds)
print(f'Silhouette Score for {n_clusters} clusters: {score:.3f}')

```

---

## Conclusions

The K-Means algorithm clustered the dataset into 5 groups with different profiles based on Annual Income and Spending Score, considering Gender as an additional categorical feature for visualization. The Elbow and Silhouette methods helped validate the choice of clusters.

---

# Repository Directory Suggestion

- README.md (this file)
- clientes-shopping.csv (dataset)
- kmeans_implementation.ipynb (Jupyter notebook with code)
- requirements.txt (for dependencies)

---

This README.md provides a complete educational and practical tutorial for K-Means clustering, balancing theory and practice with clear bilingual explanations and high-quality visualizations.

-->































<br><br><br><br>
<br><br><br><br>
<br><br><br><br>
<br><br><br><br>
<br><br><br><br>


<!-- ========================== [Bibliographr ====================  -->

<br><br>


## [Bibliography]()


[1](). **Castro, L. N. & Ferrari, D. G.** (2016). *Introdução à mineração de dados: conceitos básicos, algoritmos e aplicações*. Saraiva.

[2](). **Ferreira, A. C. P. L. et al.** (2024). *Inteligência Artificial - Uma Abordagem de Aprendizado de Máquina*. 2nd Ed. LTC.

[3](). **Larson & Farber** (2015). *Estatística Aplicada*. Pearson.


<br><br>


<!-- ======================================= Start Footer ===========================================  -->


<br><br>


## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br><br>



#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />




<br><br><br>

<p align="center">  ────────────── 🔭⋆ ──────────────


<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>

<!--
<p align="center">  ────────────── ✦ ──────────────
-->



<!-- Programmers and artists are the only professionals whose hobby is their profession."

" I love people who are committed to transforming the world "

" I'm big fan of those who are making waves in the world! "

##### <p align="center">( Rafael Lain ) </p>   -->

#

###### <p align="center"> Copyright 2025 Quantum Software Development. Code released under the [MIT License license.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)











