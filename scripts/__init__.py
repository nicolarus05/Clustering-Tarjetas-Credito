"""
Paquete de Scripts para Clustering de Tarjetas de Crédito
==========================================================

Este paquete contiene módulos reutilizables para:
- Carga y validación de datos
- Preprocesamiento y limpieza
- Aplicación de algoritmos de clustering
- Visualizaciones y análisis

Módulos:
    - carga_datos: Funciones para cargar y validar datasets
    - preprocesamiento: Limpieza, normalización y transformación
    - clustering: Algoritmos de clustering y métricas
    - visualizaciones: Gráficos y visualizaciones

Autor: [Tu Nombre]
Fecha: Enero 2026
Versión: 1.0.0
"""

__version__ = '1.0.0'
__author__ = '[Tu Nombre]'
__email__ = 'tu.email@ejemplo.com'

# Importar funciones principales de cada módulo
from scripts.carga_datos import (
    cargar_dataset,
    validar_datos,
    obtener_info_dataset
)

from scripts.preprocesamiento import (
    limpiar_datos,
    normalizar_datos,
    seleccionar_variables,
    detectar_outliers_iqr
)

from scripts.clustering import (
    calcular_inercia,
    calcular_silhouette,
    aplicar_kmeans,
    calcular_metricas_clustering
)

from scripts.visualizaciones import (
    grafico_metodo_codo,
    grafico_silhouette,
    grafico_clusters_2d,
    grafico_distribucion_clusters
)

# Lista de funciones exportadas
__all__ = [
    # Carga de datos
    'cargar_dataset',
    'validar_datos',
    'obtener_info_dataset',
    
    # Preprocesamiento
    'limpiar_datos',
    'normalizar_datos',
    'seleccionar_variables',
    'detectar_outliers_iqr',
    
    # Clustering
    'calcular_inercia',
    'calcular_silhouette',
    'aplicar_kmeans',
    'calcular_metricas_clustering',
    
    # Visualizaciones
    'grafico_metodo_codo',
    'grafico_silhouette',
    'grafico_clusters_2d',
    'grafico_distribucion_clusters'
]

# Mensaje de bienvenida al importar el paquete
print(f"📦 Paquete 'scripts' v{__version__} cargado correctamente")
