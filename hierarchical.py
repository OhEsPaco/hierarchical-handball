# -*- coding: utf-8 -*-
import numpy

import matplotlib.pyplot as plt

import pandas as pd
# Lo primero es cargar todos los datos
alldata = pd.read_excel("dataset.xlsx")
# Excluimos las columnas que no hagan falta para el clustering
exclude = ['Id', 'Time', 'scoring', 'MVP', 'Phase', 'Match', 'Team', 'No', 'Name', '%', '7m%']
# Cargamos en la variable data los datos que vamos a utilizar para hacer clustering
data = alldata.loc[:, alldata.columns.difference(exclude)]
# eliminamos valores perdidos
data.dropna()

# Se normalizan los datos
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
datanorm = min_max_scaler.fit_transform(data)

# Principal Component Analysis
from sklearn.decomposition import PCA
estimator = PCA (n_components = 2)
X_pca = estimator.fit_transform(datanorm)

plt.plot(X_pca[:,0], X_pca[:,1],'x')
plt.show()

# Matriz de similitud
import sklearn.neighbors
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(datanorm)
avSim = numpy.average(matsim)

# Dendrograma	
from scipy import cluster
clusters = cluster.hierarchy.linkage(matsim, method = 'ward')
cluster.hierarchy.dendrogram(clusters, color_threshold=20)
plt.show()

# Cortar y mostrar
cut = 20
labels = cluster.hierarchy.fcluster(clusters, cut , criterion = 'distance')
# información sobre los resultados
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print ('Numero de clusteres obtenidos: %d' % n_clusters_)
from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data, labels))

# Mostrar
colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = numpy.hstack([colors] * 20)

fig, ax = plt.subplots()
plt.xlim(-0.15, 1.5)
plt.ylim(-0.15, 1)
plt.title('Numero de clusteres obtenidos: %d' % n_clusters_)
for i in range(len(X_pca)):
    plt.text(X_pca[i][0], X_pca[i][1], i, color=colors[labels[i]])  
    
ax.grid(True)
fig.tight_layout()
plt.show()

# Creamos una variable para cada uno de los clusters donde se introducirán los elementos 
# asignados a cada cluster
alldata['group'] = labels
res_estadistica = alldata.iloc[:,2:].groupby(('group')).agg(['mean', 'max', 'min'])
res_estadistica.to_excel("clusters.xlsx")

df = alldata[['Name', 'group']]

(df.set_index('group')
 .groupby(level='group')
 .apply(lambda g: g.apply(pd.value_counts))
 .unstack(level=1)
 .fillna(0))

df.to_excel("jugadoras_clusters.xlsx")
 
