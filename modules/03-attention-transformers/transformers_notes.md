

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
