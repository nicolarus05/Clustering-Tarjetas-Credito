# 💳 Segmentación de Clientes con Tarjetas de Crédito

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Descripción del Proyecto

Este proyecto realiza **segmentación de clientes** basándose en sus patrones de uso de tarjetas de crédito, utilizando técnicas de **Machine Learning no supervisado** (K-Means Clustering).

### 🎯 Objetivos

1. Identificar grupos de clientes con comportamientos similares
2. Caracterizar cada segmento para estrategias de marketing
3. Proporcionar insights accionables para el negocio

---

## 📊 Dataset

**Fuente**: [Kaggle - Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

**Características**:
- **8,950 clientes** activos
- **18 variables** de comportamiento
- Periodo: **6 meses**

### Variables Principales

| Variable | Descripción |
|----------|-------------|
| `BALANCE` | Saldo en la cuenta |
| `PURCHASES` | Total de compras |
| `CASH_ADVANCE` | Adelantos en efectivo |
| `CREDIT_LIMIT` | Límite de crédito |
| `PAYMENTS` | Pagos realizados |
| `TENURE` | Antigüedad (meses) |

---

## 🔬 Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Estadísticas descriptivas
- Distribuciones de variables
- Análisis de correlaciones
- Detección de outliers

### 2. Preprocesamiento
- Manejo de valores nulos
- Normalización con StandardScaler
- Reducción de dimensionalidad (PCA)

### 3. Clustering
- **Método del Codo** para determinar K óptimo
- **Silhouette Score** para validación
- **K-Means Clustering**
- Visualizaciones 2D y 3D

### 4. Interpretación
- Perfiles de cada cluster
- Características distintivas
- Recomendaciones de negocio

---

## 📈 Resultados

### Clusters Identificados

#### 🟢 Cluster 0: Transactors (30%)
- Pagan el saldo completo cada mes
- Uso moderado de la tarjeta
- Bajo balance promedio
- **Estrategia**: Programas de rewards, cashback

#### 🔵 Cluster 1: Revolvers (25%)
- Mantienen balance alto
- Pagos mínimos frecuentes
- Alta generación de intereses
- **Estrategia**: Productos de consolidación de deuda

#### 🟡 Cluster 2: VIP Customers (20%)
- Alto límite de crédito
- Compras elevadas
- Pago completo consistente
- **Estrategia**: Servicios premium, upgrades

#### 🔴 Cluster 3: Cash Advance Users (25%)
- Uso frecuente de adelantos
- Bajas compras regulares
- Señal de problemas financieros
- **Estrategia**: Educación financiera, alternativas

---

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.8 o superior
- pip

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/Clustering-Tarjetas-Credito.git
cd Clustering-Tarjetas-Credito

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requisitos.txt

# Descargar el dataset
# Colocar CC_GENERAL.csv en la carpeta datos/
