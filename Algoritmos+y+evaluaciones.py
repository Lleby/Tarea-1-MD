
# coding: utf-8

# In[2]:

import pandas as pd #para el manejo de DataFrames
import numpy as np #para operaciones matemáticas

from sklearn.cluster import KMeans #importación del algoritmo de clustering k-means
from sklearn.cluster import MiniBatchKMeans #importación del algoritmo de clustering minibatch k-means
from sklearn.cluster import AgglomerativeClustering as HAC #importación del algoritmo de clustering HAC complete/ward
from sklearn.cluster import DBSCAN #importación del algoritmo de clustering DBScan

from sklearn.decomposition import PCA #para reducción de dimensionalidad por medio del algoritmo PCA
import matplotlib.pyplot as plt #para visualización de clustering obtenido

from sklearn.metrics import silhouette_score #para evaluación de clustering obtenido: índice silhouette
from sklearn.metrics.cluster import normalized_mutual_info_score #para evaluación de clustering obtenido: índice NMI
from sklearn.metrics.cluster import adjusted_rand_score #para evaluación de clustering obtenido: índice ARi
from sklearn.metrics.cluster import fowlkes_mallows_score #para evaluación de clustering obtenido: índice FMi

#se carga en reviews con relleno de registros vacíos
reviews = pd.read_csv(open('beer_reviews.csv', encoding="utf8")).fillna(method='ffill')

#estructura del dataset original
print("Dimensionalidad del dataset: ", reviews.shape[1])
print("Cantidad de registros: ", reviews.shape[0])

#tratamiento de dataset
reviews=reviews.sample(frac=0.01)
etiquetas= reviews[['beer_style','beer_beerid']]
puntajes=reviews[['review_overall','review_aroma','review_appearance','review_palate','review_taste', 'beer_abv']]

puntajes = PCA(n_components = 2).fit_transform(puntajes)

#K-MEANS

k_means = KMeans(init = "k-means++", n_clusters = 2, n_init = 10)

k_means.fit(puntajes)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

print("Coeficiente de Silhouette: ", silhouette_score(puntajes, k_means_labels))

for k in np.unique(k_means_labels):
    plt.plot(puntajes[k_means_labels == k, 0], puntajes[k_means_labels == k, 1], '.', color = plt.cm.jet(np.float(k) / np.max(k_means_labels + 1)))

plt.title("Clustering K-means")
plt.show()

#MINI BATCH K-MEANS

mini_batch = MiniBatchKMeans(init = 'k-means++', n_clusters = 2, n_init = 10, batch_size = 50)
mini_batch.fit(puntajes)

mini_batch_labels = mini_batch.labels_

print("Coeficiente de Silhouette: ", silhouette_score(puntajes, mini_batch_labels))

for k in np.unique(mini_batch_labels):
    plt.plot(puntajes[mini_batch_labels == k, 0], puntajes[mini_batch_labels == k, 1], '.', color = plt.cm.jet(np.float(k) / np.max(mini_batch_labels + 1)))

plt.title("Clustering Minibatch K-means")
plt.show()

#HAC COMPLETE

hac_complete = HAC(linkage = 'complete', n_clusters = 2, affinity = 'euclidean')
hac_complete.fit(puntajes)

hac_complete_labels = hac_complete.labels_

print("Coeficiente de Silhouette: ", silhouette_score(puntajes, hac_complete_labels))

for k in np.unique(hac_complete_labels):
    plt.plot(puntajes[hac_complete_labels == k, 0], puntajes[hac_complete_labels == k, 1], '.', color = plt.cm.jet(np.float(k) / np.max(hac_complete_labels + 1)))

plt.title("Clustering HAC complete")
plt.show()

#HAC WARD

hac_ward = HAC(linkage = 'ward', n_clusters = 2, affinity = 'euclidean')
hac_ward.fit(puntajes)

hac_ward_labels = hac_ward.labels_

print("Coeficiente de Silhouette: ", silhouette_score(puntajes, hac_ward_labels))

for k in np.unique(hac_ward_labels):
    plt.plot(puntajes[hac_ward_labels == k, 0], puntajes[hac_ward_labels == k, 1], '.', color = plt.cm.jet(np.float(k) / np.max(hac_ward_labels + 1)))

plt.title("Clustering HAC ward")
plt.show()

#DBSCAN

db = DBSCAN(eps = 0.7, min_samples = 9).fit(puntajes)
core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
db_labels = db.labels_

print("Coeficiente de Silhouette: ", silhouette_score(puntajes, db_labels))
unique_labels = set(db_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'

    class_member_mask = (db_labels == k)

    xy = puntajes[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)
    xy = puntajes[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

plt.title("DBSCAN")
plt.show()

#EVALUACIÓN DE ETIQUETAS
print("EVALUACIÓN DE ETIQUETAS\n")


id_cerveza=etiquetas.as_matrix(['beer_beerid'])
id_cerveza=np.reshape(id_cerveza, (id_cerveza.shape[0],))
print("Informe Caso 1\n")

#Informe NMI
nmi_k = normalized_mutual_info_score(k_means_labels, id_cerveza)
nmi_mini = normalized_mutual_info_score(mini_batch_labels, id_cerveza)
nmi_hac_c = normalized_mutual_info_score(hac_complete_labels, id_cerveza)
nmi_hac_w = normalized_mutual_info_score(hac_ward_labels, id_cerveza)
nmi_db = normalized_mutual_info_score(db_labels, id_cerveza)
print("NMI k_means     : ", nmi_k)
print("NMI mini batch  : ", nmi_mini)
print("NMI HAC complete: ", nmi_hac_c)
print("NMI HAC ward    : ", nmi_hac_w)
print("NMI DBScan      : ", nmi_db)
print("NMI promedio    : ", (nmi_k+ nmi_mini + nmi_hac_c + nmi_hac_w + nmi_db)/5)
print("\n")

#Informe ARI
ari_k = adjusted_rand_score(k_means_labels, id_cerveza)
ari_mini = adjusted_rand_score(mini_batch_labels, id_cerveza)
ari_hac_c = adjusted_rand_score(hac_complete_labels, id_cerveza)
ari_hac_w = adjusted_rand_score(hac_ward_labels, id_cerveza)
ari_db = adjusted_rand_score(db_labels, id_cerveza)
print("ARI k_means     : ", ari_k)
print("ARI mini batch  : ", ari_mini)
print("ARI HAC complete: ", ari_hac_c)
print("ARI HAC ward    : ", ari_hac_w)
print("ARI DBScan      : ", ari_db)
print("ARI promedio    : ", (ari_k+ ari_mini + ari_hac_c + ari_hac_w + ari_db)/5)
print("\n")

tipo_cerveza=etiquetas.as_matrix(['beer_style'])
tipo_cerveza=np.reshape(tipo_cerveza, (tipo_cerveza.shape[0],))
print("Informe Caso 2\n")

#Informe NMI
nmi_k = normalized_mutual_info_score(k_means_labels, tipo_cerveza)
nmi_mini = normalized_mutual_info_score(mini_batch_labels, tipo_cerveza)
nmi_hac_c = normalized_mutual_info_score(hac_complete_labels, tipo_cerveza)
nmi_hac_w = normalized_mutual_info_score(hac_ward_labels, tipo_cerveza)
nmi_db = normalized_mutual_info_score(db_labels, tipo_cerveza)
print("NMI k_means     : ", nmi_k)
print("NMI mini batch  : ", nmi_mini)
print("NMI HAC complete: ", nmi_hac_c)
print("NMI HAC ward    : ", nmi_hac_w)
print("NMI DBScan      : ", nmi_db)
print("NMI promedio    : ", (nmi_k+ nmi_mini + nmi_hac_c + nmi_hac_w + nmi_db)/5)
print("\n")

#Informe ARI
ari_k = adjusted_rand_score(k_means_labels, tipo_cerveza)
ari_mini = adjusted_rand_score(mini_batch_labels, tipo_cerveza)
ari_hac_c = adjusted_rand_score(hac_complete_labels, tipo_cerveza)
ari_hac_w = adjusted_rand_score(hac_ward_labels, tipo_cerveza)
ari_db = adjusted_rand_score(db_labels, tipo_cerveza)
print("ARI k_means     : ", ari_k)
print("ARI mini batch  : ", ari_mini)
print("ARI HAC complete: ", ari_hac_c)
print("ARI HAC ward    : ", ari_hac_w)
print("ARI DBScan      : ", ari_db)
print("ARI promedio    : ", (ari_k+ ari_mini + ari_hac_c + ari_hac_w + ari_db)/5)
print("\n")

#VISUALIZACION EJEMPLO CONCLUSIONES

print("VISUALIZACION DE EJEMPLO, CONCLUSIONES\n")

reviews_aux=reviews.sample(frac=0.01)
etiquetas_aux= reviews_aux[['beer_style','beer_beerid']]
puntajes_aux=reviews_aux[['review_overall','review_aroma','review_appearance','review_palate','review_taste', 'beer_abv']]
puntajes_aux = PCA(n_components = 2).fit_transform(puntajes_aux)

k_means = KMeans(init = "k-means++", n_clusters = 2, n_init = 10)

k_means.fit(puntajes_aux)
k_means_labels_aux = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

print("Coeficiente de Silhouette: ", silhouette_score(puntajes_aux, k_means_labels_aux))

for k in np.unique(k_means_labels_aux):
    plt.plot(puntajes_aux[k_means_labels_aux == k, 0], puntajes_aux[k_means_labels_aux == k, 1], '.', color = plt.cm.jet(np.float(k) / np.max(k_means_labels_aux + 1)))

plt.title("Clustering reducido K-means")
plt.show()


# In[ ]:



