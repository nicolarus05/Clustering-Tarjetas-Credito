"""
Módulo de Visualizaciones
==========================

Este módulo contiene funciones para:
- Graficar Método del Codo
- Graficar Silhouette Scores
- Visualizar clusters en 2D y 3D
- Crear heatmaps de perfiles
- Gráficos de distribución

Autor: [Tu Nombre]
Fecha: Enero 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def grafico_metodo_codo(rango_k, inercias, guardar=False, ruta='resultados/graficos/metodo_codo.png'):
    """
    Crea el gráfico del Método del Codo.
    
    Parámetros:
    -----------
    rango_k : range o list
        Valores de K probados
    inercias : list
        Lista de inercias correspondientes
    guardar : bool
        Si True, guarda el gráfico (por defecto: False)
    ruta : str
        Ruta donde guardar el gráfico
    
    Retorna:
    --------
    matplotlib.figure.Figure
        Figura del gráfico
    
    Ejemplo:
    --------
    >>> fig = grafico_metodo_codo(range(2, 11), inercias, guardar=True)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Graficar línea
    ax.plot(rango_k, inercias, marker='o', linewidth=2, markersize=10, color='steelblue')
    
    # Añadir valores en cada punto
    for k, inercia in zip(rango_k, inercias):
        ax.text(k, inercia, f'{inercia:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Línea de referencia (ejemplo K=4)
    if 4 in rango_k:
        ax.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='K=4 (ejemplo)')
    
    ax.set_title('Método del Codo - Determinación del K Óptimo', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax.set_ylabel('Inercia (Suma de Distancias al Cuadrado)', fontsize=12)
    ax.set_xticks(rango_k)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(ruta, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {ruta}")
    
    return fig


def grafico_silhouette(rango_k, silhouette_scores, guardar=False, 
                       ruta='resultados/graficos/silhouette_score.png'):
    """
    Crea el gráfico de Silhouette Scores.
    
    Parámetros:
    -----------
    rango_k : range o list
        Valores de K probados
    silhouette_scores : list
        Lista de scores correspondientes
    guardar : bool
        Si True, guarda el gráfico (por defecto: False)
    ruta : str
        Ruta donde guardar el gráfico
    
    Retorna:
    --------
    matplotlib.figure.Figure
        Figura del gráfico
    
    Ejemplo:
    --------
    >>> fig = grafico_silhouette(range(2, 11), scores, guardar=True)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Graficar línea
    ax.plot(rango_k, silhouette_scores, marker='s', linewidth=2, 
            markersize=10, color='green')
    
    # Añadir valores
    for k, score in zip(rango_k, silhouette_scores):
        ax.text(k, score, f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Marcar mejor K
    mejor_k = rango_k[np.argmax(silhouette_scores)]
    ax.axvline(x=mejor_k, color='red', linestyle='--', alpha=0.5, 
               label=f'Mejor K={mejor_k}')
    
    # Líneas de referencia de calidad
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.3, label='Bueno (>0.5)')
    ax.axhline(y=0.3, color='yellow', linestyle=':', alpha=0.3, label='Aceptable (>0.3)')
    
    ax.set_title('Silhouette Score por Número de Clusters', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Número de Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_xticks(rango_k)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(ruta, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {ruta}")
    
    return fig


def grafico_clusters_2d(df_pca, clusters, guardar=False, 
                        ruta='resultados/graficos/clusters_2d.png'):
    """
    Visualiza clusters en 2D usando componentes principales.
    
    Parámetros:
    -----------
    df_pca : pd.DataFrame
        DataFrame con componentes principales (PC1, PC2)
    clusters : np.ndarray
        Etiquetas de cluster
    guardar : bool
        Si True, guarda el gráfico (por defecto: False)
    ruta : str
        Ruta donde guardar el gráfico
    
    Retorna:
    --------
    matplotlib.figure.Figure
        Figura del gráfico
    
    Ejemplo:
    --------
    >>> fig = grafico_clusters_2d(df_pca, labels, guardar=True)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colores y marcadores
    colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    
    n_clusters = len(np.unique(clusters))
    
    # Añadir columna de cluster al dataframe temporal
    df_temp = df_pca.copy()
    df_temp['Cluster'] = clusters
    
    # Graficar cada cluster
    for i in range(n_clusters):
        cluster_data = df_temp[df_temp['Cluster'] == i]
        ax.scatter(
            cluster_data['PC1'],
            cluster_data['PC2'],
            c=colores[i % len(colores)],
            marker=markers[i % len(markers)],
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=f'Cluster {i} (n={len(cluster_data)})'
        )
        
        # Calcular y marcar centroide
        centroide_x = cluster_data['PC1'].mean()
        centroide_y = cluster_data['PC2'].mean()
        ax.scatter(
            centroide_x,
            centroide_y,
            c=colores[i % len(colores)],
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2
        )
    
    ax.set_title(f'Visualización de {n_clusters} Clusters - PCA 2D', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Componente Principal 1 (PC1)', fontsize=12)
    ax.set_ylabel('Componente Principal 2 (PC2)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(ruta, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {ruta}")
    
    return fig


def grafico_clusters_3d(df_pca, clusters, guardar=False,
                        ruta='resultados/graficos/clusters_3d.png'):
    """
    Visualiza clusters en 3D usando componentes principales.
    
    Parámetros:
    -----------
    df_pca : pd.DataFrame
        DataFrame con componentes principales (PC1, PC2, PC3)
    clusters : np.ndarray
        Etiquetas de cluster
    guardar : bool
        Si True, guarda el gráfico (por defecto: False)
    ruta : str
        Ruta donde guardar el gráfico
    
    Retorna:
    --------
    matplotlib.figure.Figure
        Figura del gráfico
    
    Ejemplo:
    --------
    >>> fig = grafico_clusters_3d(df_pca, labels, guardar=True)
    """
    if 'PC3' not in df_pca.columns:
        print("⚠️ No hay PC3 disponible para visualización 3D")
        return None
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    n_clusters = len(np.unique(clusters))
    
    df_temp = df_pca.copy()
    df_temp['Cluster'] = clusters
    
    # Graficar cada cluster
    for i in range(n_clusters):
        cluster_data = df_temp[df_temp['Cluster'] == i]
        ax.scatter(
            cluster_data['PC1'],
            cluster_data['PC2'],
            cluster_data['PC3'],
            c=colores[i % len(colores)],
            marker=markers[i % len(markers)],
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=f'Cluster {i}'
        )
        
        # Centroide
        centroide_x = cluster_data['PC1'].mean()
        centroide_y = cluster_data['PC2'].mean()
        centroide_z = cluster_data['PC3'].mean()
        ax.scatter(
            centroide_x,
            centroide_y,
            centroide_z,
            c=colores[i % len(colores)],
            marker='X',
            s=300,
            edgecolors='black',
            linewidth=2
        )
    
    ax.set_title(f'Visualización 3D de {n_clusters} Clusters', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_zlabel('PC3', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(ruta, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {ruta}")
    
    return fig


def grafico_distribucion_clusters(clusters, guardar=False,
                                   ruta='resultados/graficos/distribucion_clusters.png'):
    """
    Crea gráfico de pastel con la distribución de clusters.
    
    Parámetros:
    -----------
    clusters : np.ndarray
        Etiquetas de cluster
    guardar : bool
        Si True, guarda el gráfico (por defecto: False)
    ruta : str
        Ruta donde guardar el gráfico
    
    Retorna:
    --------
    matplotlib.figure.Figure
        Figura del gráfico
    
    Ejemplo:
    --------
    >>> fig = grafico_distribucion_clusters(labels, guardar=True)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contar clientes por cluster
    unique, counts = np.unique(clusters, return_counts=True)
    
    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    
    # Crear gráfico de pastel
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=[f'Cluster {i}\n({counts[i]} clientes)' for i in unique],
        autopct='%1.1f%%',
        colors=colores[:len(unique)],
        startangle=90,
        explode=[0.05] * len(unique),
        shadow=True,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Mejorar estilo de porcentajes
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax.set_title(f'Distribución de Clientes por Cluster (K={len(unique)})', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(ruta, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {ruta}")
    
    return fig


def grafico_heatmap_perfiles(df_medias, guardar=False,
                              ruta='resultados/graficos/perfiles_clusters.png'):
    """
    Crea heatmap de perfiles de clusters.
    
    Parámetros:
    -----------
    df_medias : pd.DataFrame
        DataFrame con medias por cluster (index=cluster, columns=variables)
    guardar : bool
        Si True, guarda el gráfico (por defecto: False)
    ruta : str
        Ruta donde guardar el gráfico
    
    Retorna:
    --------
    matplotlib.figure.Figure
        Figura del gráfico
    
    Ejemplo:
    --------
    >>> medias = df.groupby('Cluster').mean()
    >>> fig = grafico_heatmap_perfiles(medias, guardar=True)
    """
    # Normalizar por columna para ver patrones relativos
    df_norm = (df_medias - df_medias.mean()) / df_medias.std()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(
        df_norm.T,
        annot=False,
        cmap='RdYlGn',
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Valor Normalizado (z-score)'},
        ax=ax
    )
    
    ax.set_title('Perfil de Características por Cluster', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Variables', fontsize=12)
    
    plt.tight_layout()
    
    if guardar:
        plt.savefig(ruta, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {ruta}")
    
    return fig


def grafico_interactivo_plotly(df_pca, clusters):
    """
    Crea visualización interactiva con Plotly.
    
    Parámetros:
    -----------
    df_pca : pd.DataFrame
        DataFrame con componentes principales
    clusters : np.ndarray
        Etiquetas de cluster
    
    Retorna:
    --------
    plotly.graph_objects.Figure
        Figura interactiva de Plotly
    
    Ejemplo:
    --------
    >>> fig = grafico_interactivo_plotly(df_pca, labels)
    >>> fig.show()
    """
    df_temp = df_pca.copy()
    df_temp['Cluster'] = clusters
    
    fig = px.scatter(
        df_temp,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=f'Clusters Interactivos (K={len(np.unique(clusters))})',
        color_continuous_scale='Viridis',
        hover_data=['PC1', 'PC2']
    )
    
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white'))
    )
    
    fig.update_layout(
        width=900,
        height=600,
        font=dict(size=12),
        title_font_size=16
    )
    
    return fig


# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo
    from sklearn.datasets import make_blobs
    
    X, y = make_blobs(n_samples=300, centers=4, random_state=42)
    df_pca = pd.DataFrame(X, columns=['PC1', 'PC2'])
    clusters = y
    
    print("📊 Creando visualizaciones de ejemplo...\n")
    
    # Gráficos
    fig1 = grafico_metodo_codo(range(2, 11), [1000, 800, 600, 500, 450, 420, 400, 390, 385])
    fig2 = grafico_silhouette(range(2, 11), [0.3, 0.45, 0.52, 0.48, 0.42, 0.38, 0.35, 0.32, 0.30])
    fig3 = grafico_clusters_2d(df_pca, clusters)
    fig4 = grafico_distribucion_clusters(clusters)
    
    plt.show()
    
    print("✅ Visualizaciones creadas correctamente")
