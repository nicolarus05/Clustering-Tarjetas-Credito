# 🚀 Guía Rápida - Proyecto Clustering Tarjetas de Crédito

## 📋 Pasos para Ejecutar el Proyecto

### 1️⃣ **Preparar el Entorno**

```bash
# Navega al directorio del proyecto
cd Clustering-Tarjetas-Credito

# El entorno virtual ya está creado en .venv
# Actívalo:
.venv\Scripts\activate
```

### 2️⃣ **Verificar el Dataset**

✅ El dataset `CC GENERAL.csv` debe estar en la carpeta `datos/`

```
Clustering-Tarjetas-Credito/
├── datos/
│   └── CC GENERAL.csv  ← ¡Aquí debe estar!
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb
└── ...
```

### 3️⃣ **Abrir el Notebook**

```bash
# Opción 1: Jupyter Notebook
jupyter notebook notebooks/01_analisis_exploratorio.ipynb

# Opción 2: JupyterLab
jupyter lab

# Opción 3: VS Code (Recomendado)
# Simplemente abre el archivo .ipynb en VS Code
```

### 4️⃣ **Ejecutar el Análisis**

1. Abre `notebooks/01_analisis_exploratorio.ipynb`
2. Ejecuta las celdas en orden (Shift + Enter)
3. Observa los resultados y visualizaciones

---

## 📊 Estructura del Análisis

| Celda | Contenido | Tiempo aprox. |
|-------|-----------|---------------|
| 1 | Importar librerías | 5 segundos |
| 2 | Cargar dataset | 2 segundos |
| 3 | Exploración inicial | 3 segundos |
| 4 | Estadísticas descriptivas | 5 segundos |
| 5 | Distribuciones (histogramas) | 10 segundos |
| 6 | Detección de outliers | 10 segundos |
| 7 | Matriz de correlación | 15 segundos |
| 8 | Top correlaciones | 3 segundos |

**⏱️ Tiempo total estimado:** ~1 minuto

---

## 🔧 Solución de Problemas

### ❌ Error: "No se encuentra el archivo"

**Causa:** El dataset no está en la carpeta correcta

**Solución:**
```bash
# Verifica que el archivo existe
dir datos\

# Debe aparecer: CC GENERAL.csv
```

### ❌ Error: "ModuleNotFoundError: No module named 'pandas'"

**Causa:** El entorno virtual no está activado o faltan dependencias

**Solución:**
```bash
# Activa el entorno virtual
.venv\Scripts\activate

# Instala las dependencias
pip install -r requisitos.txt
```

### ❌ Error: Kernel no conectado

**Causa:** El notebook no está usando el kernel correcto

**Solución en VS Code:**
1. Haz clic en "Select Kernel" (arriba a la derecha)
2. Selecciona: "Python 3.x.x (.venv)"

---

## 📈 Próximos Pasos

Una vez completado el análisis exploratorio:

1. ✅ **Preprocesamiento** 
   - Normalización de datos
   - Manejo de valores nulos
   - Reducción de dimensionalidad (PCA)

2. ✅ **Clustering**
   - Método del codo (elbow method)
   - Aplicar K-Means
   - Validar con Silhouette Score

3. ✅ **Interpretación**
   - Perfilar cada cluster
   - Generar insights de negocio
   - Visualizar segmentos

---

## 💡 Consejos

- 🔍 **Lee las descripciones** de cada sección antes de ejecutar
- 📊 **Analiza los gráficos** detenidamente
- 💭 **Piensa en implicaciones** de negocio de cada hallazgo
- 📝 **Toma notas** de insights interesantes

---

## 📞 ¿Necesitas Ayuda?

- 📧 Email: [tu-email@ejemplo.com]
- 💬 GitHub Issues: [Link al repositorio]
- 📚 Documentación: Ver README.md

---

**¡Feliz análisis! 🎉**
