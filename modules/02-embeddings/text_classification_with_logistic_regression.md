# Fundamentos de NLP: Clasificación de Texto con Regresión Logística

> **Objetivo personal de esta nota:** Quitarle la "magia" a los modelos de lenguaje y a NLP. Entender qué conforma el pipeline por dentro sin perderme en fórmulas vacías, aterrizando exactamente qué hace cada pieza.

---

## 1. La gran duda: ¿Por qué ver esto si ya vimos Word2Vec y Tokenización?

En el capítulo anterior vimos cómo romper el texto en piezas (tokenización) y convertirlo en números/espacios continuos (vectorización, Word2Vec, etc.). 

La duda lógica fue: *¿Por qué el profesor me muestra ahora regresión logística tradicional?*

* **La respuesta:** Tokenizar y vectorizar **solo preparan el texto**. No toman decisiones.
* Una vez que tienes el texto convertido en números, necesitas un mecanismo para **clasificarlo** (por ejemplo: saber si un tweet es tóxico o no, o a qué departamento pertenece una reseña).
* Esta lámina muestra el clasificador base clásico. Aunque en modelos modernos (Transformers o LLMs) la forma de extraer características cambie, **la cabeza de decisión al final sigue haciendo exactamente esto mismo**.

---

## 2. El Pipeline explicado con mis palabras

El flujo completo que se muestra en la lámina consta de tres pasos claros:

```
Texto crudo: "I liked the cat <eos>" / "Me gusta maquillarme..."
   │
   ▼
[ 1. Extractor de características ] ──► Se genera el vector 'h' (ej. bolsa de palabras / BoW)
   │
   ▼
[ 2. Ponderación por clase (Pesos 'w') ] ──► Cada clase evalúa qué tanto le gustan esas palabras
   │
   ▼
[ 3. Softmax ] ──► Pasa los puntajes brutos a probabilidades que suman 100%
```

---

## 3. Desglosando los componentes: ¿Qué es qué?

### A. ¿Qué diablos es $h$?
**$h$ es simplemente tu texto convertido en una lista de números (vector de características).**
* En este enfoque manual (BoW / *Bag of Words*), imaginas un vocabulario fijo.
* Cada posición del vector es una palabra de ese vocabulario.
* Cuentas cuántas veces aparece cada palabra en tu frase y ese conteo es tu vector $h$:

$$
h = [h_1, h_2, \dots, h_D]^\top
$$

### B. ¿Quién conecta la característica con la clase? (Los Pesos $w_k$)
* **Mi duda inicial:** *¿Acaso Softmax es el que dice que "pelo" pertenece a "mamífero"?*
* **Aclaración clave:** **NO.** Eso lo hacen los **pesos ($w_k$)**.
  * Cada clase $k$ tiene sus propios pesos ($w_k$) y su sesgo ($b_k$).
  * Si la clase es "Mamífero" y la palabra es "pelo", el modelo le asignará a esa palabra un peso alto y positivo.
  * Hacemos el producto punto (multiplicar posición por posición y sumar):

$$
z_k = \langle w_k, h \rangle + b_k = w_k^\top h + b_k
$$

El resultado $z_k$ es un **puntaje bruto (*logit*)** para esa clase (puede dar $11.5$, $-2.3$, $0$, etc.).

### C. ¿Qué hace realmente Softmax (o Sigmoide)?
* **Sigmoide:** Se usa cuando solo tienes **dos opciones** (sí/no, spam/no-spam). Comprime un solo valor bruto para dejarlo entre $0$ y $1$.
* **Softmax:** Es la versión para **múltiples opciones (3 o más clases)**. Toma todos los puntajes brutos ($z_1, z_2, z_3$), los vuelve positivos con exponenciales ($e^z$) y los normaliza dividiendo entre la suma total:

$$
\mathcal{P}(\text{clase} = k \mid \text{texto}) = \frac{\exp(w_k^\top h + b_k)}{\sum_{c=1}^K \exp(w_c^\top h + b_c)}
$$

> **En resumen:** Los pesos deciden los puntos que merece cada categoría. Softmax solo es la calculadora final que convierte esos puntos en porcentajes limpios (que sumen 100%) para ver quién ganó.

---

## 4. Ejemplo aterrizado paso a paso

### Mi texto de prueba:
> *"Me gusta maquillarme con marcas como Nars y Dior porque mi piel es sensible"*

### Queremos clasificarlo en 3 clases:
* **Clase 1:** Belleza / Skincare
* **Clase 2:** Deportes
* **Clase 3:** Tecnología

---

### Paso 1: Obtener $h$ (Bolsa de palabras)
Vocabulario de referencia: `["maquillarme", "nars", "dior", "piel", "sensible", "balón", "pantalla"]`

Revisamos la frase y contamos:
$$h = [1, 1, 1, 1, 1, 0, 0]^\top$$

*(Esa es la barrita vertical verde $h$ de la diapositiva).*

---

### Paso 2: Multiplicar por los pesos ($w_k \cdot h + b_k$)
* **Para Belleza ($w_1$):** Palabras como *nars*, *dior*, *piel* tienen pesos positivos altos:
  $$z_1 = (1 \times 3.0) + (1 \times 2.5) + (1 \times 2.5) + (1 \times 2.0) + (1 \times 1.5) = \mathbf{11.5}$$
* **Para Deportes ($w_2$):** Esas palabras no aportan nada al deporte:
  $$z_2 = -0.9$$
* **Para Tecnología ($w_3$):** Tampoco tienen que ver con pantallas o chips:
  $$z_3 = -2.3$$

Tenemos los tres puntajes brutos: $[11.5, -0.9, -2.3]$.

---

### Paso 3: Aplicar Softmax
Softmax recibe $[11.5, -0.9, -2.3]$ y nos da las probabilidades:
* $P(\text{Clase 1: Belleza}) \approx \mathbf{99.99\%}$
* $P(\text{Clase 2: Deportes}) \approx \mathbf{0.008\%}$
* $P(\text{Clase 3: Tecnología}) \approx \mathbf{0.002\%}$

**Conclusión del modelo:** Se asigna con total seguridad a **Belleza**.

---

## 5. Cuadro comparativo rápido de apoyo

| Concepto | ¿Qué es en la práctica? | En la gráfica del profe |
| :--- | :--- | :--- |
| **$h$** | El vector numérico que representa el texto. | El bloque vertical verde de rectángulos. |
| **$w_k$** | Los pesos aprendidos para cada categoría. | Las filas horizontales de colores ($w_1, w_2, w_3$). |
| **$w_k \cdot h + b_k$** | El puntaje bruto o afinidad con esa clase. | Los círculos azules antes de Softmax. |
| **Softmax** | La función que convierte puntos en porcentaje ($0$ a $1$, suma $100\%$). | El bloque que entrega las barritas finales de probabilidad. |
## El cambio de nombre por el signo:Con signo positivo ($\max$): Se le llama Log-Likelihood (o Máxima Verosimilitud) porque buscas maximizar la probabilidad de acertar.   Con signo negativo ($\min$): Se le llama Log-Loss o Cross-Entropy porque buscas minimizar el error
---

## El gran salto: ¿Por qué dejar de "contar" palabras?

> **La duda clave:** *¿Creamos una red lo suficientemente buena para decir "ya no vamos a contar, ya sabemos el contexto"?*

**Exactamente.** Ese fue el cambio de paradigma histórico en el procesamiento de lenguaje natural (NLP):

* **El problema de contar (Bag of Words):**  
  Si comparabas:
  * *"La comida no estuvo nada rica, el servicio fue pésimo"*  
  * *"La comida no estuvo mal, el servicio fue rico"*  
  Un contador de frecuencias veía prácticamente el mismo vocabulario y se confundía con facilidad, ignorando por completo el orden y la negación.

* **La solución de Deep Learning:**  
  Se construyeron arquitecturas neuronales que leen en secuencia y consideran la posición y la relación contextual entre palabras. El sistema ya no ve frecuencias numéricas aisladas ($1, 1, 1$), sino que proyecta las palabras como vectores cargados de significado y analiza cómo interactúan entre sí (por ejemplo, cómo un *"no"* altera por completo la carga semántica del adjetivo siguiente).
  ---

# Transición a Generación de Texto y Transformers

## 1. Repaso: ¿Qué resolvimos con la Clasificación de Texto?

* **El objetivo central:** Recibir un texto completo y asignarle una única categoría o etiqueta (por ejemplo: saber si una reseña es positiva o negativa, o si un texto habla de belleza, deportes o tecnología).
* **El pipeline clásico:** Antes de Deep Learning, contabas palabras a mano (*Bag of Words* / *TF-IDF*) para armar un vector $h$, y luego una Regresión Logística (capa lineal + Softmax) calculaba la probabilidad de cada categoría evaluando el error con *Cross-Entropy*.
* **El cambio con Deep Learning:** En lugar de contar palabras a ciegas ($1, 1, 1$), una red neuronal toma vectores con significado semántico previo (*embeddings*) y aprende por sí misma a resumir la frase en el vector $h$, manteniendo al final la misma regresión logística para decidir la clase.

---

## 2. El puente intermedio: Las RNN antes de los Transformers

* **Para qué servían:** Las Redes Neuronales Recurrentes (*Recurrent Neural Networks* - RNN) fueron el primer intento exitoso de procesar secuencias palabra por palabra en orden, arrastrando un estado oculto o "memoria" paso a paso para que el modelo mantuviera el contexto de lo leído previamente.
* **Por qué quedaron obsoletas en la práctica:**
  * **Procesamiento estrictamente secuencial:** No se pueden paralelizar eficientemente en GPUs modernas, ya que para procesar el token $t$ necesitas obligatoriamente haber terminado el token $t-1$.
  * **Desvanecimiento del gradiente (*vanishing gradient*):** Sufren de pérdida de memoria crítica cuando los textos son medianamente largos.
* **Su valor pedagógico:** Sirven para entender las limitaciones del procesamiento secuencial que los **Transformers** resolvieron al procesar todos los tokens en paralelo mediante mecanismos de auto-atención (*Self-Attention*).

---

