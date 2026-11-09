

## Entrenamiento en Seq2Seq: Teacher Forcing & Loss

### 1. Inferencia vs. Entrenamiento
* **Inferencia (Greedy / Test):** No hay respuestas a mano. Lo que predice el paso $t$ entra como input al paso $t+1$ (líneas punteadas). Si se equivoca al inicio, el error se propaga en cascada.
* **Entrenamiento (Training):** Se usa **Teacher Forcing** (sin líneas punteadas). Sin importar el error que cometa el modelo, se le inyecta siempre el token correcto del corpus en el paso siguiente para que no descarrile.

### 2. Medición del error: Cross-Entropy ($J$)
En cada token $t$, el modelo predice probabilidades $\hat{y}_t$. Se penaliza según la probabilidad asignada a la palabra correcta:
$$\text{Loss}_t = -\log p(\text{token correcto})$$

El error total de la frase ($J$) es el promedio de todos los pasos:
$$J = \frac{1}{T} \sum_{t=1}^T J_t$$

> **Regla de oro:** Toda la IA generativa moderna (GPT-4, Llama, Claude) se sigue entrenando con este mismo principio: Teacher Forcing + Cross-Entropy token por token.

---

### 3. Implementación mínima

**En PyTorch:**
```python
import torch.nn as nn

# Teacher forcing: targets desplazados
logits = model(source_seq, target_inputs)

# CrossEntropyLoss calcula -log(p) y promedia en un solo paso
loss_fn = nn.CrossEntropyLoss()
J = loss_fn(logits, target_labels)
J.backward()

## Fundamentos del Transformer: De RNNs a Attention

### 1. Motivación
* **Límite de las RNNs:** Procesamiento estrictamente secuencial ($O(N)$ timesteps) que no paraleliza en GPU y crea un cuello de botella al comprimir todo en un único vector $h$[cite: 1, 5].
* **Transformer (2017):** Reemplaza la recurrencia por mecanismos puros de atención procesados simultáneamente en paralelo[cite: 1, 2].

### 2. Mecanismos Clave
* **Self-Attention ($Q, K, V$):** Cada token proyecta tres vectores (*Query*, *Key*, *Value*)[cite: 8]. La atención pondera el contenido ($V$) según la afinidad semántica entre búsqueda ($Q$) y etiquetas ($K$)[cite: 8]:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* **Multi-Head Attention:** Divide las proyecciones en múltiples cabezales independientes para capturar simultáneamente relaciones sintácticas, correferencias y dependencias semánticas[cite: 12, 13].
* **Masked Self-Attention (Decoder):** Aplica una máscara con $-\infty$ a posiciones futuras para evitar filtración de información durante el entrenamiento autorregresivo[cite: 10].
* **Encoder-Decoder (Cross-Attention):** El Decoder consulta las representaciones contextualizadas del Encoder ($Q$ desde el Decoder, $K$ y $V$ desde el Encoder)[cite: 3].
* **Positional Encoding:** Suma coordenadas a los embeddings para dotar de noción posicional a la secuencia procesada en paralelo[cite: 4].
