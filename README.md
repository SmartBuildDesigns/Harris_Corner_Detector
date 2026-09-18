# Harris Corner Detector: Teoría, Matemáticas e Implementación Interactiva

Este repositorio contiene una implementación robusta e interactiva del algoritmo de detección de esquinas propuesto por C. Harris y M. Stephens (1988). Además de la simple extracción de puntos, la herramienta incluye un dashboard analítico para visualizar el comportamiento matemático profundo del algoritmo mediante gráficas 3D de la superficie de error y proyecciones 2D del tensor de estructura.

---

## 1. Librerías Utilizadas y su Rol en el Algoritmo

Para lograr una ejecución en tiempo real y una visualización matemática interactiva, el código se apoya en el siguiente ecosistema de librerías de Python:

*   **OpenCV (`cv2`):** Es el motor principal de procesamiento de imágenes. Se utiliza para leer la imagen, convertirla a escala de grises, aplicar los filtros de derivadas de Sobel ($I_x, I_y$), realizar el suavizado direccional con filtros Gaussianos, ejecutar la dilatación morfológica para la Supresión de No Máximos (NMS) y normalizar la matriz de respuestas.
*   **NumPy (`numpy`):** Fundamental para el álgebra lineal y las operaciones vectorizadas. Evita el uso de bucles `for` lentos al calcular el Tensor de Estructura ($M$) y la respuesta $R$ en toda la matriz de la imagen simultáneamente. Además, su submódulo de álgebra lineal (`np.linalg.eigh`) es vital para extraer los eigenvalores y eigenvectores reales de las matrices de covarianza locales.
*   **Matplotlib (`matplotlib.pyplot`, `widgets`, `patches`):** Responsable de renderizar el Dashboard Interactivo. Genera la visualización 3D paramétrica de la superficie de error $E(u,v)$, dibuja la elipse del tensor utilizando `patches.Ellipse`, plotea los eigenvectores con `quiver`, y gestiona la interactividad de la interfaz (botones de navegación y el slider dinámico de umbral).
*   **Tkinter (`tkinter` y `filedialog`):** Librería nativa de interfaces gráficas de Python. Se utiliza de forma "invisible" (ocultando la ventana principal) exclusivamente para invocar el cuadro de diálogo nativo del sistema operativo, permitiendo al usuario seleccionar la imagen de entrada de forma amigable.

---

## 2. Conceptos Matemáticos Fundamentales

Para comprender cómo la computadora "ve" una esquina, es necesario definir algunas piezas clave del álgebra lineal y el procesamiento de señales:

*   **Gradiente Espacial:** Representa el cambio direccional de la intensidad (brillo) de la imagen. Si estás sobre una superficie plana de color sólido, el gradiente es cero.
*   **Eigenvectores (Vectores Propios):** En el contexto de este detector, son vectores que indican las direcciones primarias donde ocurre la mayor y menor variación de intensidad dentro de una región local de la imagen.
*   **Eigenvalores ($\lambda$):** Son magnitudes escalares asociadas a cada eigenvector. Nos indican *qué tan fuerte* es el cambio de intensidad en esa dirección específica.
    *   Si $\lambda_1$ y $\lambda_2$ son pequeños: La región es plana (flat).
    *   Si un $\lambda$ es grande y el otro pequeño: Hay un borde unidireccional (edge).
    *   Si **ambos** $\lambda_1$ y $\lambda_2$ son grandes: Hay cambios fuertes en múltiples direcciones, es decir, una esquina (corner).
*   **Determinante:** Operación matemática equivalente al producto de los eigenvalores: $\det(M) = \lambda_1 \lambda_2$.
*   **Traza:** Suma de los elementos de la diagonal principal de una matriz, equivalente a la suma de sus eigenvalores: $\text{trace}(M) = \lambda_1 + \lambda_2$.

---

## 3. El Algoritmo de Harris Paso a Paso

El núcleo del detector se basa en evaluar cómo cambia una pequeña ventana (parche) de la imagen si se desplaza en cualquier dirección $(u, v)$. Ese cambio se aproxima mediante una forma cuadrática.

### Paso 1: Cálculo de Derivadas Espaciales
Se calcula la derivada de la imagen $I$ en el eje $x$ e $y$ utilizando operadores como Sobel.

$$I_x = \frac{\partial I}{\partial x}$$

$$I_y = \frac{\partial I}{\partial y}$$

### Paso 2: Productos de las Derivadas
Calculamos el cuadrado de los gradientes y su producto cruzado para cada píxel:

$$I_{x^2} = I_x \cdot I_x$$

$$I_{y^2} = I_y \cdot I_y$$

$$I_{xy} = I_x \cdot I_y$$

### Paso 3: Función de Ventana (Suavizado Gaussiano)
Aplicamos un filtro Gaussiano ($G_\sigma$) a los productos anteriores para integrar la información del vecindario local y reducir el ruido:

$$S_{x^2} = G_\sigma * I_{x^2}$$

$$S_{y^2} = G_\sigma * I_{y^2}$$

$$S_{xy} = G_\sigma * I_{xy}$$

### Paso 4: Construcción del Tensor de Estructura ($M$)
Con los valores suavizados, definimos una matriz de covarianza de gradientes de $2 \times 2$ para cada píxel:

$$M = \begin{bmatrix} S_{x^2} & S_{xy} \\ S_{xy} & S_{y^2} \end{bmatrix}$$

### Paso 5: La Función de Respuesta ($R$)
Harris y Stephens propusieron una métrica que utiliza el determinante y la traza para estimar la presencia de una esquina sin calcular eigenvalores explícitamente:

$$R = \det(M) - k(\text{trace}(M))^2$$

*(Nota: En nuestra implementación, truncamos los valores negativos de bordes a 0 antes de normalizar la matriz al rango [0, 1] para estabilizar la interfaz gráfica).*

### Paso 6: Umbralización y Supresión de No Máximos (NMS)
1.  **Umbral (Threshold):** Se descartan todos los valores de $R$ que no superen el umbral definido por el usuario.
2.  **NMS:** Se comparan los píxeles con sus vecinos mediante dilatación morfológica, conservando solo el pico absoluto local.

---

## 4. Dashboard Interactivo y Visualizaciones

La herramienta permite explorar la anatomía matemática de cada esquina detectada:

*   **Superficie de Error Cuadrático 3D:**
    Evalúa la ecuación:
    
   $$
   E(u,v) \approx
   \begin{bmatrix} u & v \end{bmatrix}
   M
   \begin{bmatrix} u \\ v \end{bmatrix}
   $$
    
    Visualmente, una esquina fuerte genera una superficie cóncava pronunciada en forma de "cuenco".
*   **Elipse de Tensor 2D y Eigenvectores:**
    El tensor de estructura se proyecta geométricamente como una elipse descrita por $x^T M x = \text{constante}$. La orientación está dictada por los eigenvectores, y la magnitud de los ejes es proporcional a $1/\sqrt{\lambda}$.

---

## 5. Prerrequisitos y Ejecución

Para ejecutar el código, asegúrate de tener Python instalado y las siguientes librerías de terceros:

1. **Instalación de dependencias:**
   ```bash
   pip install numpy opencv-python matplotlib
2. **Ejecución:**
   ```bash
   python H_corner.py

## 6. Material de Consulta

En este repositorio también se encuentran los documentos originales en formato PDF para profundizar en la teoría y matemáticas detrás del código:

*   [**`6.2_Harris_Corner_Detector.pdf`**](6.2_Harris_Corner_Detector.pdf): Presentación académica detallando la deducción matemática y la aproximación bilineal de la superficie de error.
*   [**`Harris_Stephens_1988.pdf`**](Harris_Stephens_1988.pdf): Artículo original *A Combined Corner and Edge Detector* (Harris, C. & Stephens, M., 1988).
