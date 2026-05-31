# Resumen Ejecutivo Técnico: Proyecto de Clasificación de Enfermedades Foliares del Maíz

## 1. Definición del Problema de Machine Learning
* **Tarea de ML:** El sistema se aborda como una clasificación supervisada de imágenes, modelado como un problema multiclase de etiqueta única (*single-label classification*). Se excluyen explícitamente enfoques de detección de objetos (*object detection*) y clasificación multietiqueta (*multi-label*), ya que el objetivo es la clasificación global de la enfermedad predominante, no su ubicación espacial en la hoja.
* **Espacio de Entrada y Salida:** El modelo procesa un espacio de entrada $X \subseteq \mathbb{R}^{224 \times 224 \times 3}$ correspondiente a imágenes RGB normalizadas. El propósito es aprender una función parametrizada $f_\theta: X \to^{|Y|}$ que asigne una distribución de probabilidad sobre las clases posibles mediante una activación Softmax.
* **Clases Objetivo ($|Y|=4$):** Roya común (*Puccinia sorghi*), Tizón foliar (*Exserohilum turcicum*), Mancha gris (*Cercospora zeae-maydis*) y Hoja sana.

## 2. Estrategia de Modelado y Entrenamiento
* **Arquitectura Base:** Se prioriza el uso de la red neuronal convolucional (CNN) **MobileNetV3**, elegida por su bajo número de parámetros y viabilidad operativa en dispositivos con recursos limitados (se contrastará con arquitecturas como MobileNetV2 y EfficientNet B0-B3).
* **Transfer Learning y Adaptación de Dominio:** El entrenamiento se dividirá en tres etapas:
  1. Inicialización de la red con pesos preentrenados de ImageNet.
  2. *Fine-tuning* con imágenes de laboratorio controladas, aplicando técnicas de aumento de datos (*Data Augmentation*) como rotaciones e inversiones para mejorar la generalización.
  3. *Fine-tuning* final utilizando exclusivamente imágenes reales de campo, mitigando el sesgo (*Domain Shift*) hacia imágenes controladas.
* **Optimización:** Se empleará el optimizador Adam por su rápida convergencia y la entropía cruzada categórica (*Categorical Cross-Entropy*) como función de pérdida, aplicando una ponderación de clases para contrarrestar el desbalance natural de los datos.

## 3. Métricas de Evaluación y Criterios de Éxito
* **Métrica Principal (Recall):** Debido a la asimetría de costos en el dominio agrícola (donde el costo de un falso negativo resulta en omisión de tratamiento y daño a la cosecha), se prioriza fuertemente maximizar el **Recall** en las clases patológicas.
* **Métrica Secundaria:** Se utilizará el **Macro F1-score** para establecer un equilibrio entre precisión y sensibilidad en el entorno multiclase desbalanceado.
* **Meta de Rendimiento:** El modelo debe alcanzar un Recall y Macro F1-score $\geq 0.85$ medido sobre un conjunto de pruebas ciego e independiente, compuesto prioritariamente por imágenes reales obtenidas en campo. 

## 4. Aplicación de Usuario y Despliegue (Edge AI)
* **Interfaz y Framework:** Se desarrollará una **Aplicación Web Progresiva (PWA)** utilizando Vue 3 en el frontend, diseñada específicamente para teléfonos con ecosistema Android.
* **Inferencia Offline en el Dispositivo:** La aplicación garantizará un funcionamiento completamente offline, realizando la captura de la imagen y la inferencia directamente en el hardware del usuario sin requerir conexión a internet en las parcelas.
* **Optimización y Rendimiento del Modelo:** El modelo se convertirá a formato **TensorFlow Lite (.tflite)** utilizando cuantización post-entrenamiento a Int8. Debe cumplir estrictos criterios de hardware:
  * Tamaño del modelo empaquetado: $\leq 20$ MB.
  * Latencia de inferencia: $\leq 300$ milisegundos en teléfonos de gama media/baja (CPU equivalente a Snapdragon serie 6xx o MediaTek Helio G80 y $\geq 4$GB de RAM).
  * Consumo de RAM adicional durante ejecución: $< 200$ MB.
* **Sincronización y Backend:** Se contará con un módulo asíncrono secundario respaldado por FastAPI y MySQL que, al detectar conexión a internet, subirá imágenes comprimidas y metadatos hacia el servidor para alimentar repositorios futuros.

## 5. Diagnóstico y Recomendaciones al Productor
* **Resultado Visual:** Tras procesar la hoja, la aplicación mostrará en pantalla la clase predicha junto a su nivel de confianza probabilística.
* **Recomendaciones Oficiales:** La interfaz exhibirá recomendaciones agronómicas generales fundamentadas estrictamente en la *Guía Técnica del Cultivo de Maíz* publicada por el CENTA.
* **Responsabilidad y Advertencias (Descargo Legal):** El sistema desplegará de forma obligatoria advertencias de limitación diagnóstica, indicando claramente que actúa como una herramienta probabilística de apoyo, que no emite prescripciones químicas vinculantes y que **no sustituye la validación de un profesional técnico agrónomo**.

## 6. Metodología y Gestión de Datos
* **Metodología de Proyecto:** El flujo del desarrollo está regido por las seis fases del estándar de la industria **CRISP-DM**.
* **Conjunto de Datos Objetivo:** Se construirá un repositorio documentado y curado consolidando un mínimo de **3,200 imágenes**, balanceado de forma que contenga un aproximado de 700 capturas por cada una de las cuatro clases contempladas.

## 7. Alcances y Limitaciones (Qué se hará y qué no)
Para establecer claridad frente a expectativas excesivas técnicas o agronómicas, rigen las siguientes restricciones:
* **Enfoque Botánico:** Estricto al maíz (*Zea mays*). No se evalúan otros cultivos, malezas, plagas entomológicas (insectos) o deficiencias nutricionales del suelo.
* **Localización de Lesiones (Fuera de alcance):** No se implementarán cajas delimitadoras (*bounding boxes*) en las hojas; el diagnóstico es una etiqueta global por imagen.
* **On-Device Learning (Fuera de alcance):** La arquitectura actual recopilará información en un servidor mediante la API, pero no automatizará el reentrenamiento en vivo de la red neuronal dentro de los dispositivos de los usuarios.
* **Ecosistema Móvil:** Queda fuera del alcance el desarrollo nativo o adaptación y validación del sistema para dispositivos iOS de Apple.

## 8. Usuarios Finales e Impacto Esperado
El proyecto se enfoca en resolver una problemática crítica de seguridad alimentaria y democratización tecnológica para los siguientes usuarios directos:
* **Pequeños y Medianos Productores (Agricultores de subsistencia):** Representan el grupo beneficiario principal de la PWA. El sistema les brindará una herramienta gratuita y rápida, que funciona de manera autónoma (offline) en entornos rurales. Su propósito es apoyar la detección temprana de enfermedades para proteger sus cosechas y evitar gastos innecesarios o erróneos en agroquímicos, lo que resulta crítico debido al estrecho margen económico de estos agricultores.
* **Técnicos Extensionistas Agrícolas:** Profesionales de campo que, durante sus visitas, pueden utilizar la aplicación móvil como un sistema de validación rápida o de triaje preliminar en las parcelas, optimizando el tiempo y fortaleciendo sus diagnósticos visuales.
