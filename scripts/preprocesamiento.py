"""
Módulo de Preprocesamiento de Datos
====================================

Este módulo contiene funciones para:
- Limpiar datos (nulos, duplicados)
- Normalizar y estandarizar variables
- Detectar y tratar outliers
- Seleccionar variables relevantes

Autor: [Tu Nombre]
Fecha: Enero 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats


def limpiar_datos(df, eliminar_nulos=True, eliminar_duplicados=True, 
                  columnas_eliminar=['CUST_ID']):
    """
    Limpia el dataset eliminando nulos, duplicados y columnas innecesarias.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a limpiar
    eliminar_nulos : bool
        Si True, imputa valores nulos con la mediana (por defecto: True)
    eliminar_duplicados : bool
        Si True, elimina registros duplicados (por defecto: True)
    columnas_eliminar : list
        Lista de columnas a eliminar (por defecto: ['CUST_ID'])
    
    Retorna:
    --------
    pd.DataFrame
        Dataset limpio
    dict
        Diccionario con estadísticas de limpieza
    
    Ejemplo:
    --------
    >>> df_limpio, stats = limpiar_datos(df)
    >>> print(f"Registros eliminados: {stats['registros_eliminados']}")
    """
    df_limpio = df.copy()
    filas_iniciales = len(df_limpio)
    
    # Estadísticas de limpieza
    estadisticas = {
        'filas_iniciales': filas_iniciales,
        'columnas_iniciales': len(df_limpio.columns),
        'nulos_iniciales': df_limpio.isnull().sum().sum(),
        'duplicados_iniciales': df_limpio.duplicated().sum()
    }
    
    print("🧹 INICIANDO LIMPIEZA DE DATOS")
    print("="*70)
    
    # 1. Eliminar columnas innecesarias
    columnas_existentes = [col for col in columnas_eliminar if col in df_limpio.columns]
    if columnas_existentes:
        df_limpio = df_limpio.drop(columnas_existentes, axis=1)
        print(f"✅ Columnas eliminadas: {columnas_existentes}")
        estadisticas['columnas_eliminadas'] = columnas_existentes
    
    # 2. Manejar valores nulos
    if eliminar_nulos and df_limpio.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='median')
        columnas = df_limpio.columns
        df_limpio = pd.DataFrame(
            imputer.fit_transform(df_limpio),
            columns=columnas
        )
        print(f"✅ Valores nulos imputados con la mediana: {estadisticas['nulos_iniciales']}")
    
    # 3. Eliminar duplicados
    if eliminar_duplicados and estadisticas['duplicados_iniciales'] > 0:
        df_limpio = df_limpio.drop_duplicates()
        print(f"✅ Duplicados eliminados: {estadisticas['duplicados_iniciales']}")
    
    # Estadísticas finales
    estadisticas['filas_finales'] = len(df_limpio)
    estadisticas['columnas_finales'] = len(df_limpio.columns)
    estadisticas['registros_eliminados'] = filas_iniciales - len(df_limpio)
    estadisticas['nulos_finales'] = df_limpio.isnull().sum().sum()
    
    print(f"\n📊 Resultado:")
    print(f"   Filas: {estadisticas['filas_iniciales']:,} → {estadisticas['filas_finales']:,}")
    print(f"   Columnas: {estadisticas['columnas_iniciales']} → {estadisticas['columnas_finales']}")
    print(f"   Nulos: {estadisticas['nulos_iniciales']} → {estadisticas['nulos_finales']}")
    print("="*70 + "\n")
    
    return df_limpio, estadisticas


def normalizar_datos(df, metodo='standard'):
    """
    Normaliza o estandariza las variables del dataset.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a normalizar
    metodo : str
        Método de normalización:
        - 'standard': StandardScaler (media=0, std=1)
        - 'minmax': MinMaxScaler (rango [0, 1])
        Por defecto: 'standard'
    
    Retorna:
    --------
    pd.DataFrame
        Dataset normalizado
    scaler : StandardScaler o MinMaxScaler
        Objeto scaler ajustado (para transformar nuevos datos)
    
    Ejemplo:
    --------
    >>> df_norm, scaler = normalizar_datos(df, metodo='standard')
    >>> print(df_norm.mean())  # Debe ser ~0
    """
    # Seleccionar scaler
    if metodo == 'standard':
        scaler = StandardScaler()
        nombre_metodo = "StandardScaler (media=0, std=1)"
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
        nombre_metodo = "MinMaxScaler (rango [0,1])"
    else:
        raise ValueError(f"Método no válido: {metodo}. Usa 'standard' o 'minmax'")
    
    print(f"📏 NORMALIZANDO DATOS CON {nombre_metodo}")
    print("="*70)
    
    # Aplicar normalización
    datos_normalizados = scaler.fit_transform(df)
    
    df_normalizado = pd.DataFrame(
        datos_normalizados,
        columns=df.columns,
        index=df.index
    )
    
    # Verificar normalización
    if metodo == 'standard':
        media_promedio = df_normalizado.mean().mean()
        std_promedio = df_normalizado.std().mean()
        print(f"✅ Normalización completada")
        print(f"   Media promedio: {media_promedio:.6f} (esperado: ~0)")
        print(f"   Std promedio: {std_promedio:.6f} (esperado: ~1)")
    else:
        min_valor = df_normalizado.min().min()
        max_valor = df_normalizado.max().max()
        print(f"✅ Normalización completada")
        print(f"   Valor mínimo: {min_valor:.6f} (esperado: 0)")
        print(f"   Valor máximo: {max_valor:.6f} (esperado: 1)")
    
    print("="*70 + "\n")
    
    return df_normalizado, scaler


def detectar_outliers_iqr(df, columna=None, factor=1.5):
    """
    Detecta outliers usando el método IQR (Rango Intercuartílico).
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a analizar
    columna : str, optional
        Columna específica a analizar. Si es None, analiza todas
    factor : float
        Factor multiplicador del IQR (por defecto: 1.5)
    
    Retorna:
    --------
    pd.DataFrame o dict
        Si columna especificada: DataFrame booleano
        Si columna=None: Diccionario con resumen por columna
    
    Ejemplo:
    --------
    >>> outliers = detectar_outliers_iqr(df, columna='BALANCE')
    >>> print(f"Outliers detectados: {outliers.sum()}")
    """
    if columna is not None:
        # Análisis de una columna específica
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - factor * IQR
        limite_superior = Q3 + factor * IQR
        
        outliers = (df[columna] < limite_inferior) | (df[columna] > limite_superior)
        
        return outliers
    
    else:
        # Análisis de todas las columnas numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        
        resumen = []
        for col in columnas_numericas:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            limite_inferior = Q1 - factor * IQR
            limite_superior = Q3 + factor * IQR
            
            outliers = ((df[col] < limite_inferior) | (df[col] > limite_superior)).sum()
            porcentaje = (outliers / len(df)) * 100
            
            resumen.append({
                'Variable': col,
                'N_Outliers': outliers,
                '%_Outliers': round(porcentaje, 2),
                'Limite_Inferior': round(limite_inferior, 2),
                'Limite_Superior': round(limite_superior, 2)
            })
        
        return pd.DataFrame(resumen).sort_values('N_Outliers', ascending=False)


def seleccionar_variables(df, umbral_correlacion=0.95, metodo='correlacion'):
    """
    Selecciona variables eliminando las altamente correlacionadas.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset con variables a seleccionar
    umbral_correlacion : float
        Umbral de correlación para eliminar variables (por defecto: 0.95)
    metodo : str
        Método de selección: 'correlacion' o 'varianza'
    
    Retorna:
    --------
    list
        Lista de columnas seleccionadas
    pd.DataFrame
        Matriz de correlación (si metodo='correlacion')
    
    Ejemplo:
    --------
    >>> columnas_selec, corr = seleccionar_variables(df, umbral_correlacion=0.95)
    >>> df_reducido = df[columnas_selec]
    """
    print(f"🎯 SELECCIÓN DE VARIABLES (Método: {metodo})")
    print("="*70)
    
    if metodo == 'correlacion':
        # Calcular matriz de correlación
        corr_matrix = df.corr().abs()
        
        # Obtener triángulo superior
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Encontrar variables altamente correlacionadas
        columnas_eliminar = [
            column for column in upper.columns 
            if any(upper[column] > umbral_correlacion)
        ]
        
        columnas_seleccionadas = [
            col for col in df.columns 
            if col not in columnas_eliminar
        ]
        
        print(f"✅ Variables originales: {len(df.columns)}")
        print(f"✅ Variables eliminadas por correlación > {umbral_correlacion}: {len(columnas_eliminar)}")
        if columnas_eliminar:
            print(f"   Variables eliminadas: {columnas_eliminar}")
        print(f"✅ Variables seleccionadas: {len(columnas_seleccionadas)}")
        print("="*70 + "\n")
        
        return columnas_seleccionadas, corr_matrix
    
    elif metodo == 'varianza':
        # Eliminar variables con varianza cero o muy baja
        varianzas = df.var()
        columnas_seleccionadas = varianzas[varianzas > 1e-10].index.tolist()
        
        print(f"✅ Variables originales: {len(df.columns)}")
        print(f"✅ Variables con varianza ~0 eliminadas: {len(df.columns) - len(columnas_seleccionadas)}")
        print(f"✅ Variables seleccionadas: {len(columnas_seleccionadas)}")
        print("="*70 + "\n")
        
        return columnas_seleccionadas, None
    
    else:
        raise ValueError(f"Método no válido: {metodo}")


def tratar_outliers(df, metodo='cap', factor=1.5):
    """
    Trata outliers usando diferentes métodos.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset con outliers
    metodo : str
        Método de tratamiento:
        - 'cap': Limita valores a los límites IQR (winsorizing)
        - 'eliminar': Elimina filas con outliers
        - 'zscore': Elimina outliers por z-score > 3
    factor : float
        Factor IQR (por defecto: 1.5)
    
    Retorna:
    --------
    pd.DataFrame
        Dataset con outliers tratados
    
    Ejemplo:
    --------
    >>> df_tratado = tratar_outliers(df, metodo='cap')
    """
    df_tratado = df.copy()
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    
    print(f"🔧 TRATAMIENTO DE OUTLIERS (Método: {metodo})")
    print("="*70)
    
    if metodo == 'cap':
        # Limitar valores a los límites IQR
        for col in columnas_numericas:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            limite_inferior = Q1 - factor * IQR
            limite_superior = Q3 + factor * IQR
            
            df_tratado[col] = df[col].clip(lower=limite_inferior, upper=limite_superior)
        
        print(f"✅ Outliers limitados a rangos IQR")
    
    elif metodo == 'eliminar':
        # Eliminar filas con outliers
        filas_iniciales = len(df_tratado)
        
        for col in columnas_numericas:
            outliers = detectar_outliers_iqr(df_tratado, col, factor)
            df_tratado = df_tratado[~outliers]
        
        filas_eliminadas = filas_iniciales - len(df_tratado)
        print(f"✅ Filas eliminadas: {filas_eliminadas} ({filas_eliminadas/filas_iniciales*100:.2f}%)")
    
    elif metodo == 'zscore':
        # Eliminar por z-score
        filas_iniciales = len(df_tratado)
        z_scores = np.abs(stats.zscore(df_tratado[columnas_numericas]))
        df_tratado = df_tratado[(z_scores < 3).all(axis=1)]
        
        filas_eliminadas = filas_iniciales - len(df_tratado)
        print(f"✅ Filas eliminadas (z-score>3): {filas_eliminadas} ({filas_eliminadas/filas_iniciales*100:.2f}%)")
    
    else:
        raise ValueError(f"Método no válido: {metodo}")
    
    print(f"   Dimensiones finales: {df_tratado.shape}")
    print("="*70 + "\n")
    
    return df_tratado


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos de ejemplo
    from carga_datos import cargar_dataset
    
    df = cargar_dataset('../datos/CC_GENERAL.csv')
    
    if df is not None:
        # Limpiar datos
        df_limpio, stats = limpiar_datos(df)
        
        # Normalizar
        df_norm, scaler = normalizar_datos(df_limpio)
        
        # Detectar outliers
        outliers_resumen = detectar_outliers_iqr(df_limpio)
        print("\n📊 Resumen de Outliers:")
        print(outliers_resumen)
