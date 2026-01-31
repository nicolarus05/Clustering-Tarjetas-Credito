# 💳 Segmentación de Clientes con Tarjetas de Crédito

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
  <img src="https://img.shields.io/badge/Clusters-4-blue" alt="Clusters">
  <img src="https://img.shields.io/badge/Dataset-8950%20clientes-yellowgreen" alt="Dataset">
</div>

---

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema de segmentación de clientes** utilizando técnicas de **Machine Learning no supervisado** para analizar el comportamiento de usuarios de tarjetas de crédito. El objetivo es identificar grupos de clientes con patrones similares y proporcionar insights accionables para estrategias de marketing y gestión de riesgo.

### 🎯 Objetivos

1. **Identificar segmentos** de clientes con comportamientos similares
2. **Caracterizar cada grupo** para estrategias de marketing personalizadas
3. **Proporcionar insights de negocio** accionables
4. **Optimizar la gestión** de productos y servicios financieros

---

## 📊 Dataset

**Fuente**: [Kaggle - Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

### Características del Dataset

- **📈 Registros**: 8,950 clientes activos
- **📊 Variables**: 18 características de comportamiento
- **⏱️ Periodo**: 6 meses de actividad
- **🌍 Alcance**: Datos reales anonimizados

### Variables Principales

| Variable | Descripción | Tipo |
|----------|-------------|------|
| `CUST_ID` | ID único del cliente | Identificador |
| `BALANCE` | Saldo en la cuenta | Numérico |
| `PURCHASES` | Total de compras | Numérico |
| `ONEOFF_PURCHASES` | Compras únicas | Numérico |
| `INSTALLMENTS_PURCHASES` | Compras a plazos | Numérico |
| `CASH_ADVANCE` | Adelantos en efectivo | Numérico |
| `CREDIT_LIMIT` | Límite de crédito | Numérico |
| `PAYMENTS` | Pagos realizados | Numérico |
| `MINIMUM_PAYMENTS` | Pagos mínimos | Numérico |
| `PRC_FULL_PAYMENT` | % pago completo | Numérico (0-1) |
| `TENURE` | Antigüedad (meses) | Numérico |

---

## 🔬 Metodología

### 1️⃣ Análisis Exploratorio de Datos (EDA)

- Estadísticas descriptivas completas
- Análisis de distribuciones
- Matriz de correlaciones
- Detección de outliers
- Identificación de patrones

### 2️⃣ Preprocesamiento

```python
✅ Eliminación de columnas innecesarias (CUST_ID)
✅ Imputación de valores nulos con la mediana
✅ Eliminación de registros duplicados
✅ Normalización con StandardScaler
✅ Reducción de dimensionalidad (PCA)
