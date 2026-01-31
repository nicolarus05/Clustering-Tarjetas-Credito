"""
Módulo de Carga y Validación de Datos
======================================

Este módulo contiene funciones para:
- Cargar datasets desde archivos CSV
- Validar la calidad de los datos
- Generar reportes de información del dataset

Autor: [Tu Nombre]
Fecha: Enero 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path


def cargar_dataset(ruta='datos/CC_GENERAL.csv', encoding='utf-8'):
    """
    Carga un dataset desde un archivo CSV.
    
    Parámetros:
    -----------
    ruta : str
        Ruta al archivo CSV (por defecto: 'datos/CC_GENERAL.csv')
    encoding : str
        Codificación del archivo (por defecto: 'utf-8')
    
    Retorna:
    --------
    pd.DataFrame
        Dataset cargado como DataFrame de pandas
    
    Excepciones:
    ------------
    FileNotFoundError : Si el archivo no existe
    pd.errors.EmptyDataError : Si el archivo está vacío
    
    Ejemplo:
    --------
    >>> df = cargar_dataset('datos/CC_GENERAL.csv')
    >>> print(df.shape)
    (8950, 18)
    """
    try:
        # Verificar que el archivo existe
        ruta_path = Path(ruta)
        if not ruta_path.exists():
            raise FileNotFoundError(f"❌ Archivo no encontrado: {ruta}")
        
        # Cargar el dataset
        df = pd.read_csv(ruta, encoding=encoding)
        
        # Validar que no está vacío
        if df.empty:
            raise pd.errors.EmptyDataError(f"❌ El archivo está vacío: {ruta}")
        
        print(f"✅ Dataset cargado exitosamente")
        print(f"   📊 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print(f"   💾 Memoria: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        return df
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"❌ Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error inesperado al cargar datos: {str(e)}")
        return None


def validar_datos(df, min_registros=10, min_columnas=2):
    """
    Valida la calidad y estructura del dataset.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a validar
    min_registros : int
        Número mínimo de registros requeridos (por defecto: 10)
    min_columnas : int
        Número mínimo de columnas requeridas (por defecto: 2)
    
    Retorna:
    --------
    dict
        Diccionario con el reporte de validación conteniendo:
        - es_valido: bool
        - dimensiones: tuple
        - nulos_totales: int
        - duplicados: int
        - columnas_numericas: int
        - errores: list
        - advertencias: list
    
    Ejemplo:
    --------
    >>> reporte = validar_datos(df)
    >>> if reporte['es_valido']:
    ...     print("Dataset válido")
    """
    errores = []
    advertencias = []
    
    # Validar que el dataframe no es None
    if df is None:
        return {
            'es_valido': False,
            'errores': ['DataFrame es None']
        }
    
    # Validar dimensiones mínimas
    if len(df) < min_registros:
        errores.append(f"Registros insuficientes: {len(df)} < {min_registros}")
    
    if len(df.columns) < min_columnas:
        errores.append(f"Columnas insuficientes: {len(df.columns)} < {min_columnas}")
    
    # Contar nulos
    nulos_totales = df.isnull().sum().sum()
    if nulos_totales > 0:
        pct_nulos = (nulos_totales / (len(df) * len(df.columns))) * 100
        advertencias.append(f"Valores nulos detectados: {nulos_totales} ({pct_nulos:.2f}%)")
    
    # Contar duplicados
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        pct_duplicados = (duplicados / len(df)) * 100
        advertencias.append(f"Registros duplicados: {duplicados} ({pct_duplicados:.2f}%)")
    
    # Verificar columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(columnas_numericas) == 0:
        errores.append("No hay columnas numéricas en el dataset")
    
    # Verificar varianza en columnas numéricas
    if len(columnas_numericas) > 0:
        varianzas_cero = (df[columnas_numericas].var() == 0).sum()
        if varianzas_cero > 0:
            advertencias.append(f"Columnas con varianza cero: {varianzas_cero}")
    
    # Determinar si el dataset es válido
    es_valido = len(errores) == 0
    
    # Crear reporte
    reporte = {
        'es_valido': es_valido,
        'dimensiones': df.shape,
        'nulos_totales': int(nulos_totales),
        'duplicados': int(duplicados),
        'columnas_numericas': len(columnas_numericas),
        'errores': errores,
        'advertencias': advertencias,
        'memoria_mb': df.memory_usage(deep=True).sum() / (1024**2)
    }
    
    # Imprimir reporte
    print("\n" + "="*70)
    print("🔍 REPORTE DE VALIDACIÓN")
    print("="*70)
    print(f"Estado: {'✅ VÁLIDO' if es_valido else '❌ NO VÁLIDO'}")
    print(f"Dimensiones: {df.shape[0]:,} × {df.shape[1]}")
    print(f"Valores nulos: {nulos_totales:,}")
    print(f"Duplicados: {duplicados}")
    print(f"Columnas numéricas: {len(columnas_numericas)}")
    
    if errores:
        print(f"\n❌ Errores ({len(errores)}):")
        for error in errores:
            print(f"   • {error}")
    
    if advertencias:
        print(f"\n⚠️ Advertencias ({len(advertencias)}):")
        for adv in advertencias:
            print(f"   • {adv}")
    
    print("="*70 + "\n")
    
    return reporte


def obtener_info_dataset(df):
    """
    Genera información detallada del dataset por columna.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con información detallada de cada columna:
        - Tipo: tipo de dato
        - No_Nulos: cantidad de valores no nulos
        - Nulos: cantidad de valores nulos
        - %_Nulos: porcentaje de nulos
        - Valores_Unicos: cantidad de valores únicos
        - Media: media (solo numéricas)
        - Min: valor mínimo (solo numéricas)
        - Max: valor máximo (solo numéricas)
    
    Ejemplo:
    --------
    >>> info = obtener_info_dataset(df)
    >>> print(info)
    """
    info_dict = {
        'Tipo': df.dtypes,
        'No_Nulos': df.count(),
        'Nulos': df.isnull().sum(),
        '%_Nulos': (df.isnull().sum() / len(df) * 100).round(2),
        'Valores_Unicos': df.nunique()
    }
    
    info_df = pd.DataFrame(info_dict)
    
    # Añadir estadísticas para columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    info_df['Media'] = np.nan
    info_df['Min'] = np.nan
    info_df['Max'] = np.nan
    
    for col in columnas_numericas:
        info_df.loc[col, 'Media'] = df[col].mean()
        info_df.loc[col, 'Min'] = df[col].min()
        info_df.loc[col, 'Max'] = df[col].max()
    
    return info_df


def exportar_reporte(df, ruta_salida='resultados/reporte_datos.txt'):
    """
    Exporta un reporte completo del dataset a un archivo de texto.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a reportar
    ruta_salida : str
        Ruta donde guardar el reporte
    
    Retorna:
    --------
    bool
        True si el reporte se guardó exitosamente
    
    Ejemplo:
    --------
    >>> exportar_reporte(df, 'resultados/reporte_datos.txt')
    """
    try:
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE DATOS - CLUSTERING TARJETAS DE CRÉDITO\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Fecha de generación: {pd.Timestamp.now()}\n")
            f.write(f"Registros: {len(df):,}\n")
            f.write(f"Columnas: {len(df.columns)}\n")
            f.write(f"Memoria: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\n\n")
            
            f.write("COLUMNAS:\n")
            f.write("-"*80 + "\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i:2d}. {col}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✅ Reporte exportado: {ruta_salida}")
        return True
        
    except Exception as e:
        print(f"❌ Error al exportar reporte: {str(e)}")
        return False


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar dataset
    df = cargar_dataset('../datos/CC_GENERAL.csv')
    
    if df is not None:
        # Validar datos
        reporte = validar_datos(df)
        
        # Obtener información
        info = obtener_info_dataset(df)
        print("\n📊 Información del dataset:")
        print(info)
        
        # Exportar reporte
        exportar_reporte(df)
