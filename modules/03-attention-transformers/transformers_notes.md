

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
# Aspect-Based Sentiment Classification with Transformers (TripAdvisor Reviews)

Implementación y evaluación comparativa de modelos basados en arquitectura Transformer para clasificación de sentimiento por aspectos (*Aspect-Based Sentiment Analysis*) sobre reseñas de hoteles en inglés.

---

## 📌 Resumen de la Práctica

En esta práctica se evaluaron 4 etapas clave de fine-tuning y optimización:

1. **Preparación y Mapeo de Datos:**
   - **Aspecto objetivo:** `cleanliness` (limpieza), filtrando registros sin calificación.
   - **Discretización (3 clases):** Mapeo de escala 1–5 a polaridades `BAD` (1–2), `NEUTRAL` (3) y `GOOD` (4–5).
   - **División:** Split estratificado (*train / validation / test*) preservando la distribución original y tokenización con truncamiento a 256 tokens.

2. **Fine-Tuning y Comparación de Modelos:**
   - **Baseline:** `bert-base-uncased` con *Early Stopping*.
   - **Comparador:** `roberta-base` bajo el mismo esquema de entrenamiento y métrica de evaluación (*Macro F1*).
   - **Veredicto:** RoBERTa superó a BERT gracias a su tokenizador BPE por bytes y eliminación de la tarea NSP, capturando mejor expresiones coloquiales y reseñas ambiguas.

3. **Optimización de Hiperparámetros (Optuna):**
   - Ajuste sistemático de *learning rate* ($1\times 10^{-5}$ a $5\times 10^{-5}$) y *batch size* ({16, 32}).
   - El modelo reentrenado con los parámetros óptimos incrementó el *Macro F1* en el conjunto de prueba, reduciendo el error en la clase minoritaria (`NEUTRAL`).

4. **Ablación de Capas (Frozen Backbone vs. Full Fine-Tuning):**
   - Congelamiento de todas las capas del encoder excepto la última (capa 11) y el cabezal de clasificación.
   - **Hallazgo:** Entrenar únicamente ~6.5% de los parámetros redujo el tiempo de entrenamiento a la mitad, pero disminuyó el desempeño en ~6–8% de F1. Esto demuestra la necesidad de adaptar pesos intermedios para alinear términos de sentimiento con aspectos específicos.

---

## 🛠️ Tecnologías Utilizadas
- **Frameworks:** PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`
- **Optimización:** Optuna
- **Métricas:** Scikit-Learn (`accuracy`, `macro_f1`)
- **Entorno:** Google Colab (GPU T4)

---

## 💻 Comandos Git para Guardar Cambios

```bash
git add . && git commit -m "feat(nlp): complete bert & roberta aspect-based sentiment classification homework" && git push
