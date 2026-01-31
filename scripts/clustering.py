"""
Módulo de Clustering
====================

Este módulo contiene funciones para:
- Calcular inercias (Método del Codo)
- Calcular Silhouette Scores
- Aplicar algoritmos de clustering (K-Means, DBSCAN, etc.)
- Calcular métricas de evaluación

Autor: [Tu Nombre]
Fecha: Enero 2026
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples
)
from sklearn.decomposition import PCA


def calcular_inercia(df, rango_k=range(2, 11), random_state=42):
    """
    Calcula la inercia para diferentes valores de K (Método del Codo).
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset normalizado
    rango_k : range
        Rango de valores de K a probar (por defecto: 2 a 10)
    random_state : int
        Semilla aleatoria para reproducibilidad (por defecto: 42)
    
    Retorna:
    --------
    list
        Lista de inercias para cada K
    dict
        Diccionario con K como clave e inercia como valor
    
    Ejemplo:
    --------
    >>> inercias, dict_inercias = calcular_inercia(df_normalizado)
    >>> print(f"Inercia para K=3: {dict_inercias[3]}")
    """
    print("🔄 CALCULANDO INERCIAS (MÉTODO DEL CODO)")
    print("="*70)
    
    inercias = []
    dict_inercias = {}
    
    for k in rango_k:
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        kmeans.fit(df)
        inercia = kmeans.inertia_
        inercias.append(inercia)
        dict_inercias[k] = inercia
        
        print(f"K={k:2d} → Inercia: {inercia:>12,.2f}")
    
    print("="*70)
    print("✅ Cálculo de inercias completado\n")
    
    return inercias, dict_inercias


def calcular_silhouette(df, rango_k=range(2, 11), random_state=42):
    """
    Calcula el Silhouette Score para diferentes valores de K.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset normalizado
    rango_k : range
        Rango de valores de K a probar (por defecto: 2 a 10)
    random_state : int
        Semilla aleatoria (por defecto: 42)
    
    Retorna:
    --------
    list
        Lista de Silhouette Scores para cada K
    dict
        Diccionario con K como clave y score como valor
    int
        K óptimo (mayor silhouette score)
    
    Ejemplo:
    --------
    >>> scores, dict_scores, mejor_k = calcular_silhouette(df_normalizado)
    >>> print(f"Mejor K: {mejor_k} con score {dict_scores[mejor_k]:.4f}")
    """
    print("🔄 CALCULANDO SILHOUETTE SCORES")
    print("="*70)
    
    silhouette_scores = []
    dict_scores = {}
    
    for k in rango_k:
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(df)
        score = silhouette_score(df, labels)
        silhouette_scores.append(score)
        dict_scores[k] = score
        
        print(f"K={k:2d} → Silhouette Score: {score:.4f}")
    
    # Identificar mejor K
    mejor_k = max(dict_scores, key=dict_scores.get)
    mejor_score = dict_scores[mejor_k]
    
    print("="*70)
    print(f"✅ Mejor K: {mejor_k} (Score: {mejor_score:.4f})\n")
    
    return silhouette_scores, dict_scores, mejor_k


def aplicar_kmeans(df, n_clusters=4, random_state=42, verbose=True):
    """
    Aplica el algoritmo K-Means clustering.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset normalizado
    n_clusters : int
        Número de clusters (por defecto: 4)
    random_state : int
        Semilla aleatoria (por defecto: 42)
    verbose : bool
        Si True, imprime información (por defecto: True)
    
    Retorna:
    --------
    KMeans
        Modelo entrenado
    np.ndarray
        Array con las etiquetas de cluster para cada registro
    dict
        Diccionario con información del modelo
    
    Ejemplo:
    --------
    >>> modelo, labels, info = aplicar_kmeans(df_normalizado, n_clusters=4)
    >>> print(f"Inercia: {info['inercia']}")
    """
    if verbose:
        print(f"🎯 APLICANDO K-MEANS CON K={n_clusters}")
        print("="*70)
    
    # Entrenar modelo
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
        algorithm='lloyd'
    )
    
    labels = kmeans.fit_predict(df)
    
    # Información del modelo
    info = {
        'n_clusters': n_clusters,
        'inercia': kmeans.inertia_,
        'iteraciones': kmeans.n_iter_,
        'centroides': kmeans.cluster_centers_,
        'labels': labels
    }
    
    if verbose:
        print(f"✅ Modelo entrenado exitosamente")
        print(f"   Inercia: {info['inercia']:,.2f}")
        print(f"   Iteraciones: {info['iteraciones']}")
        print(f"   Centroides: {n_clusters}")
        
        # Distribución de clusters
        print(f"\n📊 Distribución de clusters:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            porcentaje = (count / len(labels)) * 100
            print(f"   Cluster {cluster}: {count:>5} ({porcentaje:>5.2f}%)")
        
        print("="*70 + "\n")
    
    return kmeans, labels, info


def calcular_metricas_clustering(df, labels, verbose=True):
    """
    Calcula múltiples métricas de evaluación del clustering.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset normalizado
    labels : np.ndarray
        Etiquetas de cluster asignadas
    verbose : bool
        Si True, imprime las métricas (por defecto: True)
    
    Retorna:
    --------
    dict
        Diccionario con todas las métricas:
        - silhouette_score: Medida de separación entre clusters
        - davies_bouldin_index: Compacidad y separación (menor es mejor)
        - calinski_harabasz_score: Ratio dispersión inter/intra (mayor es mejor)
        - n_clusters: Número de clusters
    
    Ejemplo:
    --------
    >>> metricas = calcular_metricas_clustering(df_normalizado, labels)
    >>> print(f"Silhouette Score: {metricas['silhouette_score']:.4f}")
    """
    # Calcular métricas
    silhouette = silhouette_score(df, labels)
    davies_bouldin = davies_bouldin_score(df, labels)
    calinski = calinski_harabasz_score(df, labels)
    n_clusters = len(np.unique(labels))
    
    metricas = {
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_score': calinski,
        'n_clusters': n_clusters
    }
    
    if verbose:
        print("📊 MÉTRICAS DE EVALUACIÓN DEL CLUSTERING")
        print("="*70)
        print(f"Número de clusters:       {n_clusters}")
        print(f"Silhouette Score:         {silhouette:.4f}  (Rango: [-1, 1], óptimo: ~1)")
        print(f"Davies-Bouldin Index:     {davies_bouldin:.4f}  (Menor es mejor)")
        print(f"Calinski-Harabasz Score:  {calinski:.2f}  (Mayor es mejor)")
        print("="*70)
        
        # Interpretación del Silhouette Score
        if silhouette > 0.7:
            print("✅ Excelente: Clusters muy bien separados")
        elif silhouette > 0.5:
            print("✅ Bueno: Clusters bien separados")
        elif silhouette > 0.3:
            print("⚠️ Moderado: Separación aceptable")
        else:
            print("❌ Pobre: Clusters poco separados")
        
        print()
    
    return metricas


def aplicar_dbscan(df, eps=0.5, min_samples=5, verbose=True):
    """
    Aplica el algoritmo DBSCAN (clustering basado en densidad).
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset normalizado
    eps : float
        Radio de vecindad (por defecto: 0.5)
    min_samples : int
        Número mínimo de puntos para formar cluster (por defecto: 5)
    verbose : bool
        Si True, imprime información (por defecto: True)
    
    Retorna:
    --------
    DBSCAN
        Modelo entrenado
    np.ndarray
        Array con las etiquetas de cluster (-1 indica ruido)
    dict
        Diccionario con información del modelo
    
    Ejemplo:
    --------
    >>> modelo, labels, info = aplicar_dbscan(df_normalizado, eps=0.5)
    >>> print(f"Clusters encontrados: {info['n_clusters']}")
    """
    if verbose:
        print(f"🎯 APLICANDO DBSCAN (eps={eps}, min_samples={min_samples})")
        print("="*70)
    
    # Entrenar modelo
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df)
    
    # Información del modelo
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = list(labels).count(-1)
    
    info = {
        'n_clusters': n_clusters,
        'n_ruido': n_ruido,
        'labels': labels,
        'eps': eps,
        'min_samples': min_samples
    }
    
    if verbose:
        print(f"✅ Modelo entrenado exitosamente")
        print(f"   Clusters encontrados: {n_clusters}")
        print(f"   Puntos de ruido: {n_ruido} ({n_ruido/len(labels)*100:.2f}%)")
        
        # Distribución de clusters
        if n_clusters > 0:
            print(f"\n📊 Distribución de clusters:")
            unique, counts = np.unique(labels[labels != -1], return_counts=True)
            for cluster, count in zip(unique, counts):
                porcentaje = (count / len(labels)) * 100
                print(f"   Cluster {cluster}: {count:>5} ({porcentaje:>5.2f}%)")
        
        print("="*70 + "\n")
    
    return dbscan, labels, info


def aplicar_clustering_jerarquico(df, n_clusters=4, linkage='ward', verbose=True):
    """
    Aplica clustering jerárquico (Agglomerative Clustering).
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset normalizado
    n_clusters : int
        Número de clusters (por defecto: 4)
    linkage : str
        Método de enlace: 'ward', 'complete', 'average', 'single'
        (por defecto: 'ward')
    verbose : bool
        Si True, imprime información (por defecto: True)
    
    Retorna:
    --------
    AgglomerativeClustering
        Modelo entrenado
    np.ndarray
        Array con las etiquetas de cluster
    dict
        Diccionario con información del modelo
    
    Ejemplo:
    --------
    >>> modelo, labels, info = aplicar_clustering_jerarquico(df_normalizado)
    >>> print(f"Clusters: {info['n_clusters']}")
    """
    if verbose:
        print(f"🎯 APLICANDO CLUSTERING JERÁRQUICO (K={n_clusters}, linkage={linkage})")
        print("="*70)
    
    # Entrenar modelo
    modelo = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage
    )
    labels = modelo.fit_predict(df)
    
    # Información del modelo
    info = {
        'n_clusters': n_clusters,
        'linkage': linkage,
        'labels': labels
    }
    
    if verbose:
        print(f"✅ Modelo entrenado exitosamente")
        
        # Distribución de clusters
        print(f"\n📊 Distribución de clusters:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            porcentaje = (count / len(labels)) * 100
            print(f"   Cluster {cluster}: {count:>5} ({porcentaje:>5.2f}%)")
        
        print("="*70 + "\n")
    
    return modelo, labels, info


def obtener_estadisticas_clusters(df_original, labels):
    """
    Calcula estadísticas descriptivas por cluster.
    
    Parámetros:
    -----------
    df_original : pd.DataFrame
        Dataset original (sin normalizar)
    labels : np.ndarray
        Etiquetas de cluster asignadas
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con estadísticas (mean, median, std) por cluster
    
    Ejemplo:
    --------
    >>> stats = obtener_estadisticas_clusters(df_original, labels)
    >>> print(stats.loc['mean'])
    """
    # Añadir columna de cluster
    df_con_clusters = df_original.copy()
    df_con_clusters['Cluster'] = labels
    
    # Calcular estadísticas por cluster
    estadisticas = df_con_clusters.groupby('Cluster').agg(['mean', 'median', 'std', 'min', 'max'])
    
    return estadisticas


def predecir_cluster(modelo, df_nuevo, scaler=None):
    """
    Predice el cluster para nuevos datos.
    
    Parámetros:
    -----------
    modelo : KMeans (o similar)
        Modelo de clustering entrenado
    df_nuevo : pd.DataFrame
        Nuevos datos a clasificar
    scaler : StandardScaler, optional
        Scaler para normalizar (si los datos no están normalizados)
    
    Retorna:
    --------
    np.ndarray
        Array con las etiquetas de cluster predichas
    
    Ejemplo:
    --------
    >>> nuevos_clusters = predecir_cluster(modelo, df_nuevos, scaler)
    >>> print(f"Cluster asignado: {nuevos_clusters[0]}")
    """
    # Normalizar si se proporciona scaler
    if scaler is not None:
        df_nuevo_norm = pd.DataFrame(
            scaler.transform(df_nuevo),
            columns=df_nuevo.columns
        )
    else:
        df_nuevo_norm = df_nuevo
    
    # Predecir
    clusters = modelo.predict(df_nuevo_norm)
    
    return clusters


def aplicar_pca_clustering(df, n_components=2, n_clusters=4, random_state=42):
    """
    Aplica PCA seguido de K-Means clustering.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset normalizado
    n_components : int
        Número de componentes principales (por defecto: 2)
    n_clusters : int
        Número de clusters (por defecto: 4)
    random_state : int
        Semilla aleatoria (por defecto: 42)
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con componentes principales
    np.ndarray
        Etiquetas de cluster
    PCA
        Modelo PCA ajustado
    KMeans
        Modelo K-Means ajustado
    
    Ejemplo:
    --------
    >>> df_pca, labels, pca, kmeans = aplicar_pca_clustering(df_normalizado)
    >>> print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")
    """
    print(f"🎯 APLICANDO PCA + K-MEANS")
    print("="*70)
    
    # Aplicar PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    datos_pca = pca.fit_transform(df)
    
    df_pca = pd.DataFrame(
        datos_pca,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    varianza_explicada = pca.explained_variance_ratio_.sum()
    print(f"✅ PCA aplicado: {n_components} componentes")
    print(f"   Varianza explicada: {varianza_explicada*100:.2f}%")
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(df_pca)
    
    print(f"✅ K-Means aplicado: {n_clusters} clusters")
    print(f"   Inercia: {kmeans.inertia_:.2f}")
    print("="*70 + "\n")
    
    return df_pca, labels, pca, kmeans


# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo
    from sklearn.datasets import make_blobs
    
    X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    
    print("📊 Datos de ejemplo generados\n")
    
    # Calcular inercias
    inercias, dict_inercias = calcular_inercia(df)
    
    # Calcular silhouette scores
    scores, dict_scores, mejor_k = calcular_silhouette(df)
    
    # Aplicar K-Means
    modelo, labels, info = aplicar_kmeans(df, n_clusters=4)
    
    # Calcular métricas
    metricas = calcular_metricas_clustering(df, labels)
    
    print("✅ Ejemplo ejecutado correctamente")
