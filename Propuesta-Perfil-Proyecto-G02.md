**UNIVERSIDAD DE EL SALVADOR**  
**FACULTAD DE INGENIERÍA Y ARQUITECTURA**  
**ESCUELA DE INGENIERÍA DE SISTEMAS INFORMÁTICOS**  
**CURSO DE ESPECIALIZACIÓN EN MACHINE LEARNING**  
**CICLO I 2026**

![UES][image1]

**PROPUESTA DE PERFIL DE PROYECTO:**

Detección de Enfermedades Foliares en Cultivos de Maíz mediante Deep Learning en Dispositivos Móviles

**GRUPO DE TRABAJO**  
02

**DOCENTE**  
Ing. Bladimir Diaz Campos

**PRESENTADO POR:** 

| Apellido | Nombre | Carnet |
| :---- | :---- | :---- |
| Rivas Fuentes | Josías Abner | RF20010 |
| Deras Cerros | David Alejandro | DC19019 |
| Rosales Molina | Elmer Edenilson | RM20001 |

Ciudad Universitaria, 18 de abril de 2026

**Contenido**

[Planteamiento del problema	1](#planteamiento-del-problema)

[Relevancia social	2](#relevancia-social)

[Objetivos	3](#objetivos)

[Objetivo general	3](#objetivo-general)

[Objetivos específicos	3](#objetivos-específicos)

[Justificación e impacto esperado	4](#justificación-e-impacto-esperado)

[Marco teórico	4](#marco-teórico)

[Redes Neuronales Convolucionales (CNN)	4](#redes-neuronales-convolucionales-\(cnn\))

[Transfer Learning	5](#transfer-learning)

[Función de perdida	5](#función-de-perdida)

[Optimización del modelo	6](#optimización-del-modelo)

[Métricas de Evaluación	6](#métricas-de-evaluación)

[Arquitectura del modelo	6](#arquitectura-del-modelo)

[Despliegue del modelo	7](#despliegue-del-modelo)

[Metodología	7](#metodología)

[Comprensión del Negocio	7](#comprensión-del-negocio)

[Comprensión de los datos	8](#comprensión-de-los-datos)

[Preparación de los datos	9](#preparación-de-los-datos)

[Modelado	9](#modelado)

[Evaluación	10](#evaluación)

[Despliegue y distribución	10](#despliegue-y-distribución)

[Cronograma	12](#cronograma)

[Recursos necesarios	13](#recursos-necesarios)

[Recursos Humanos	13](#recursos-humanos)

[Recursos Computacionales	13](#recursos-computacionales)

[Recursos de Software	13](#recursos-de-software)

[Recursos de Datos	14](#recursos-de-datos)

[Infraestructura de Despliegue	14](#infraestructura-de-despliegue)

[Recursos Económicos	14](#recursos-económicos)

[Riesgos	14](#riesgos)

[Bibliografía	15](#bibliografía)

[Anexos	16](#anexos)

[Recopilación de datasets	16](#recopilación-de-datasets)

# **Planteamiento del problema** {#planteamiento-del-problema}

La agricultura en El Salvador es un pilar socioeconómico fundamental para la seguridad alimentaria y el desarrollo rural, representado aproximadamente el 5.6% del Producto Interno Bruto (PIB). El ecosistema agrícola salvadoreño se distingue por una marcada fragmentación de la tenencia de la tierra y una alta concentración de pequeños productores. Los registros censales agropecuarios recientes establecen la existencia de aproximadamente 395,588 productores a nivel nacional. De este total, 325,044 individuos (representando el 82.1% de la demografía agrícola) se clasifican como pequeños productores, muchos de los cuales operan a nivel de agricultura familiar o de estricta subsistencia.

Dentro de este sector, el cultivo del maíz forma parte de la economía agropecuaria y constituye uno de los principales alimentos de la dieta salvadoreña y una fuente clave de ingresos para pequeños y medianos productores. Sin embargo, la producción de maíz se ve afectada por diversos factores, entre los cuales destacan las enfermedades y plagas que reducen negativamente el rendimiento del cultivo. De acuerdo con la Organización de las Naciones Unidas para la Alimentación y la Agricultura (FAO), hasta un 40% de los cultivos a nivel mundial se pierden debido a plagas y enfermedades, lo cual refleja la magnitud del problema también en el contexto salvadoreño.

A nivel nacional, los agricultores enfrentan dificultades para identificar de manera temprana las enfermedades que afectan sus cultivos de maíz, ya que el diagnóstico suele basarse en la observación visual y en conocimiento empírico. Esta situación se agrava en zonas rurales donde el acceso a asistencia técnica especializada es limitado, lo que provoca que las enfermedades sean detectadas en etapas avanzadas, reduciendo la productividad y generando pérdidas económicas significativas.

En los últimos años, el acceso a las tecnologías de la información y comunicación (TIC) se ha vuelto atractivo para el sector agropecuario para mejorar la eficiencia y productividad. Aunque la adopción de estas tecnologías es limitada, especialmente para pequeños productores se considera el creciente acceso a teléfonos inteligentes a nivel nacional y el avance de la inteligencia artificial, particularmente en el campo de la visión por computadora que ha permitido el desarrollo de modelos capaces de identificar enfermedades en plantas a partir de imágenes. A partir de estas novedades, surge la oportunidad de desarrollar soluciones accesibles que permitan a los agricultores identificar enfermedades en sus cultivos de maíz mediante el uso de imágenes.

Las enfermedades foliares del maíz, como los tizones y las royas, presentan síntomas visuales que pueden confundirse con otros factores como estrés hídrico, daño por plagas o deficiencias nutricionales, dificultando la toma de decisiones oportunas. En este contexto, surge la necesidad de desarrollar un sistema basado en Machine Learning que permita clasificar enfermedades en cultivos de maíz a partir de imágenes, incorporando además recomendaciones generales que apoyen la toma de decisiones del agricultor.

Adicionalmente, se propone una visión de sistema evolutivo, implementado mediante herramientas portátiles, accesibles y funcionales incluso en entornos con conectividad limitada. Este sistema permitirá la recolección de nuevas imágenes en condiciones reales de uso, con el objetivo de facilitar futuras mejoras del modelo mediante procesos de reentrenamiento con datos locales. Este enfoque busca contribuir a mitigar la limitada disponibilidad de datasets abiertos en el contexto salvadoreño, cuya escasez o fragmentación puede afectar la robustez de modelos entrenados exclusivamente con imágenes obtenidas en condiciones controladas.

# **Relevancia social** {#relevancia-social}

El desarrollo de este proyecto adquiere importancia debido a su relevancia en el sector agrícola salvadoreño, especialmente en los pequeños productores que dependen del cultivo de maíz como principal fuente de sustento.

Contar con herramientas accesibles que faciliten el reconocimiento de enfermedades contribuye a mejorar las condiciones de producción, favoreciendo una gestión más eficiente de los cultivos. Asimismo, este tipo de soluciones puede apoyar la reducción de pérdidas, el aprovechamiento adecuado de recursos y la mejora en la calidad de las cosechas.

En conjunto, el proyecto se orienta a fortalecer las capacidades de los productores, contribuyendo al desarrollo del sector agropecuario y al bienestar de las comunidades vinculadas a esta actividad.

# **Objetivos** {#objetivos}

## **Objetivo general** {#objetivo-general}

Desarrollar e implementar un sistema móvil basado en aprendizaje profundo capaz de clasificar cuatro clases foliares del maíz (tres enfermedades comunes y hoja sana) a partir de imágenes capturadas con teléfonos inteligentes, optimizado para ejecutarse localmente en dispositivos Android de gama media/baja, con inferencia completamente offline y funcionalidades opcionales de sincronización en línea.

## **Objetivos específicos** {#objetivos-específicos}

1. Identificar, evaluar y consolidar datasets públicos de imágenes de hojas de maíz en condiciones controladas y de campo real, construyendo un repositorio curado de al menos 3,200 imágenes distribuidas en 4 clases, con un mínimo aproximado de 700 imágenes por clase, documentando licencias y restricciones de uso, procedencia y dominio de captura, criterios de limpieza y estrategias para mitigar desbalance.  
2. Entrenar un modelo liviano de clasificación multiclase basado en una arquitectura liviana, utilizando transferencia de aprendizaje estableciendo como meta un Macro F1-score ≥ 0.85 en un conjunto de prueba completamente independiente y compuesto prioritariamente por imágenes reales de campo. La evaluación incluirá matriz de confusión y curvas Precision-Recall por clase.  
3. Convertir el modelo entrenado a TensorFlow Lite aplicando cuantización post-entrenamiento (Int8), logrando un tamaño del modelo de ≤ 20 MB y una latencia objetivo de ≤ 300 ms por inferencia en un dispositivo Android de gama media/baja (≥4GB RAM, CPU Snapdragon serie 6xx o equivalente).  
4. Implementar una PWA que permita capturar o seleccionar imágenes, ejecutar inferencia local sin conexión a internet y mostrar la clase predicha junto con su nivel de confianza.  
5. Incorporar un módulo opcional que permita almacenar y sincronizar imágenes junto con metadatos mínimos (clase, fecha y ubicación aproximada bajo consentimiento), documentando un protocolo de gobernanza y uso futuro de los datos, sin incluir reentrenamiento dentro del alcance del presente proyecto.

# **Justificación e impacto esperado** {#justificación-e-impacto-esperado}

El maíz es el pilar de la seguridad alimentaria salvadoreña, sosteniendo a más de 2 millones de personas rurales con una producción de 17.1 millones de quintales en 376,733 manzanas. Sin embargo, el rendimiento es muy bajo (45.5 quintales / manzana) y enfermedades foliares agresivas pueden destruir hasta el 70% de la cosecha sin una detección en las primeras dos semanas. El sector enfrenta su peor crisis en una década: en 2023 la cosecha cayó un tercio en comparación con 2021 debido a factores climáticos y altos costos de insumos. Los pequeños productores operan con un margen de error económico casi nulo, donde comprar el agroquímico equivocado o actuar tarde resulta devastador. Ante la falta de asistencia técnica inmediata, es imperativo implementar una solución tecnológica descentralizada. El despliegue de una herramienta basada en Edge AI no es solo un avance tecnológico, sino una medida urgente y de bajo costo para proteger la economía de los agricultores de subsistencia mediante diagnósticos precisos y oportunos.

# **Marco teórico** {#marco-teórico}

## **Redes Neuronales Convolucionales (CNN)** {#redes-neuronales-convolucionales-(cnn)}

Las Redes Neuronales Convolucionales (CNN) son una arquitectura de Deep Learning diseñada específicamente para el procesamiento de datos con estructura espacial, como las imágenes. Su principal fortaleza radica en la capacidad de aprender automáticamente representativas jerárquicas a partir de los datos sin necesidad de ingeniería manual de características.

Las imágenes digitales pueden representarse como matrices tridimensionales:  
I(x,y)Rhwc

donde h corresponde a la altura, w al ancho y c al número de canales (por ejemplo, RGB).

Las CNN aplican filtros (kernels) que recorren la imagen para detectar patrones relevantes mediante la operación de convolución:

y(i,j)=mnx(i+m,j+n)⋅w(m,n)

x: Es la matriz de entrada (por ejemplo, los píxeles de una imagen).  
w: Es el filtro o kernel (los pesos que la red aprende).  
y(i,j): Es el valor resultante en la posición (i, j) de la capa de salida.  
m, n: Son los índices que recorren las dimensiones del filtro.

A través de múltiples capas, la red aprende diferentes niveles de abstracción: **Capas iniciales**, para detección de bordes y texturas. **Capas intermedias**, para formas y estructuras y **Capas profundas** para patrones complejos del dominio   
En el contexto agrícola, estas características incluyen los bordes y contornos de hojas, texturas superficiales, patrones necróticos y las distribuciones de manchas foliares.

## **Transfer Learning** {#transfer-learning}

El Transfer Learning es una técnica que permite aprovechar modelos previamente entrenados en grandes conjuntos de datos, como ImageNet, para resolver problemas específicos.

En este enfoque, se reutilizan los pesos de un modelo pre-entrenado y se ajustan sus últimas capas al nuevo dominio:

θ=pretrained+

Esto permite que el modelo conserve conocimiento general sobre patrones visuales, como bordes y formas, y lo adapte a características específicas como enfermedades en hojas.

## **Función de perdida** {#función-de-perdida}

La función de pérdida mide el error entre las predicciones del modelo y los valores reales. Una de ellas es **Categorical Cross-Entropy**, adecuada para problemas de clasificación multiclase:

L=-i=1nyilog⁡(yi)

Esta función penaliza fuertemente las predicciones incorrectas con alta confianza, favoreciendo una mejor calibración del modelo.

Adicionalmente, se implementa una ponderación de clases para abordar el desbalance del dataset, asignando mayor importancia a clases minoritarias, lo cual mejora la detección de enfermedades menos frecuentes.

## **Optimización del modelo** {#optimización-del-modelo}

El entrenamiento del modelo se realiza mediante algoritmos de optimización que ajustan los parámetros para minimizar la función de pérdida.

Uno de los métodos más utilizados es el gradiente descendente:  
θ=θ-αJ(θ)

Adam es otra técnica y está construido sobre el gradiente descendiente, combina técnicas de momento y adaptación de la tasa de aprendizaje, permitiendo una convergencia más rápida y estable, permitiendo un entrenamiento más rápido, fluido y fiable, especialmente para el aprendizaje profundo.

## **Métricas de Evaluación** {#métricas-de-evaluación}

Para evaluar el desempeño del modelo se utilizan las siguientes métricas:  
Accuracy=TP+TNTP+TN+FP+FNPrecision=TPTP+FPRecall=TPTP+FNF1=2⋅Precision⋅RecallPrecision+Recall

En este proyecto se prioriza el Recall, debido a la importancia de minimizar falsos negativos en la detección de enfermedades. La no detección de una enfermedad puede tener consecuencias críticas en la producción agrícola.  
El uso del F1-Score permite equilibrar precisión y sensibilidad, especialmente en datasets desbalanceados.

## **Arquitectura del modelo** {#arquitectura-del-modelo}

La arquitectura del modelo constituye la estructura interna que define la organización y conexión de las capas que conforman una red neuronal. Arquitecturas eficientes como MobileNetV3, diseñadas para operar en dispositivos con recursos computacionales limitados, permiten utilizar técnicas como Transfer Learning.  La estructura del modelo comprende una base convolucional preentrenada para la extracción de características, seguida de capas densas personalizadas para la adaptación al problema específico, y una capa de salida con función de activación Softmax para la clasificación multiclase. Este enfoque permite mantener un equilibrio entre precisión predictiva y eficiencia computacional.

## **Despliegue del modelo** {#despliegue-del-modelo}

El despliegue de modelos de aprendizaje profundo requiere una arquitectura que garantice eficiencia y tiempos de respuesta adecuados. En este proyecto se adopta un enfoque híbrido, utilizando FastAPI como backend para la gestión de solicitudes e inferencia, y un modelo optimizado con TensorFlow Lite para su ejecución eficiente. TensorFlow Lite permite reducir el tamaño del modelo y mejorar la latencia, facilitando su implementación en dispositivos móviles.

Esta arquitectura permite flujos como:

1. Captura de imagen / input  
2. Preprocesamiento  
3. Inferencia mediante CNN  
4. Generación de predicción  
5. Respuesta al usuario

# **Metodología** {#metodología}

La gestión y ejecución del proyecto se desarrollará bajo la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining). Gracias al enfoque iterativo la metodología permitirá conforme se identifiquen mejoras, alinear las decisiones técnicas del modelo con las necesidades reales del entorno agrícola y las restricciones de hardware móvil.

## **Comprensión del Negocio** {#comprensión-del-negocio}

El maíz es uno de los cultivos más importantes en El Salvador, tanto para consumo interno como para la economía de pequeños y medianos productores.

Las enfermedades foliares como la roya común, mancha gris o el tizón foliar pueden reducir la producción si no se detectan de forma temprana, y es que en la práctica el diagnóstico suele depender de la propia experiencia empírica del agricultor, recomendaciones informales y tal vez visitas técnicas del MAG u organizaciones locales, esto genera diagnósticos tardíos o incorrectos, lo que impacta en pérdidas económicas y uso ineficiente de insumos agrícolas.

El desarrollo de una herramienta basada en visión por computadora permitiría:

- Reducir el tiempo de diagnóstico.   
- Apoyar la toma de decisiones tempranas.   
- Disminuir pérdidas productivas.   
- Promover uso más racional de fungicidas.   
- Democratizar acceso a apoyo técnico mediante tecnología móvil.

En esta fase se investigarán cuáles son las clases (enfermedades) que más impactan a la planta en el país, permitiendo en las fases posteriores centrarnos en ellas. Para evaluar el éxito se utilizarán métricas estándares de clasificación: Accuracy (exactitud general), Precision por clase (para analizar falsos positivos), Recall por clase (para no omitir enfermedades reales), F1-score y la Matriz de confusión.

De forma técnica se tiene como restricciones el uso prioritario en dispositivos Android de gama media o baja, la posibilidad de uso en zonas con conectividad intermitente y la dependencia de disponibilidad de imágenes reales.

## **Comprensión de los datos** {#comprensión-de-los-datos}

En esta fase se analizarán las fuentes de datos disponibles, se realizará una consolidación multi-fuente y análisis exploratorio, priorizando datasets con fotografías de campo y fotos en entornos “de estudio” tratadas mediante técnicas como data augmentation (revisar listado inicial en [Recopilación de datasets](#recopilación-de-datasets))

Han sido considerados como fuentes repositorios académicos como Kaggle, Zenodo, Mendeley Data y repositorios institucionales, registrando y analizando por dataset:

- Licencia y permisos de uso.  
- Tipo de imagen (laboratorio vs campo).  
- Etiquetas (definición de clase)  
- Distribución por clase y posibles sesgos.

Se priorizarán imágenes en entornos reales, pero se contempla emplear datasets de imágenes en entornos de estudio como auxiliares de entrenamiento inicial para robustecer el aprendizaje de patrones generales de enfermedad, siempre y cuando la evaluación final se realice sobre datos reales.

Se delimitará inicialmente la investigación a 4 clases (3 enfermedades y hoja sana), pero de ser posible, según el dataset que se logre recopilar, se aumentará a la mayor cantidad de imágenes posibles (mientras esto no afecte la limitante de potencia por la inferencia en dispositivos móviles)

## **Preparación de los datos** {#preparación-de-los-datos}

Esta fase comprenderá:

- Limpieza y eliminación de imágenes corruptas o mal etiquetadas.  
- Estandarización del tamaño de entrada a 224x224 píxeles.  
- Normalización de valores de intensidad.  
- División del dataset en entrenamiento, validación y prueba, garantizando que el conjunto de prueba sea completamente independiente y compuesto prioritariamente por imágenes reales de campo.

Se aplicarán técnicas de aumento de datos (Data Augmentation) durante el entrenamiento, tales como:

- Rotaciones aleatorias.  
- Inversiones horizontales.  
- Recortes aleatorios.  
- Ajustes moderados de brillo y contraste.

Estas transformaciones buscarán mejorar la robustez del modelo frente a variaciones reales de iluminación, ángulo y condiciones ambientales. Existen datasets con imágenes ya tratadas, por lo que se analizará su integración ahorrando tiempo en la preparación.

## **Modelado** {#modelado}

Se empleará una estrategia de transferencia de aprendizaje utilizando la arquitectura MobileNetV3, aunque se considerarán y compararán otras arquitecturas como MobileNetV2 y EfficientNet (B0–B3), MobileNetV3 se priorizará por su bajo número de parámetros, menor consumo computacional y compatibilidad con entornos móviles. Aquí se llevará a cabo un proceso de tres etapas:

1. Inicialización con pesos preentrenados en ImageNet.  
2. Ajuste fino (fine-tuning) con imágenes en condiciones controladas (tratadas mediante data augmentation), para permitir la adaptación al dominio agrícola.  
3. Ajuste fino final con imágenes reales de campo para adaptación de dominio, favoreciendo la adaptación y reduciendo el sesgo hacia imágenes controladas.

El entrenamiento se realizará utilizando el optimizador Adam, debido a su capacidad de convergencia rápida, se definirán hiperparámetros como el learning rate inicial, tamaño de lotes según disponibilidad de memoria y la función de perdida (categorical cross-entropy)

## **Evaluación** {#evaluación}

La evaluación final se realizará sobre un conjunto de prueba independiente no utilizado durante entrenamiento ni validación. Se reportarán: Macro F1-score (métrica principal), Matriz de confusión, Precisión y Recall por clase, Curvas Precision-Recall por clase. 

Si es posible, se realizará una comparación entre las predicciones del modelo y el diagnóstico manual realizado por un agricultor con experiencia o un técnico agrícola. Este análisis permitirá estimar el nivel de concordancia entre modelo y diagnóstico humano. Adicionalmente, se evaluará el tamaño final del modelo y la latencia de inferencia en dispositivo físico de referencia (≥4GB RAM, CPU Snapdragon serie 6xx o equivalente).

El modelo se considerará viable si:

- Mantiene un Macro F1-score competitivo y equilibrado entre clases.   
- Presenta tiempos de inferencia adecuados para uso práctico en campo.   
- Demuestra coherencia razonable con el diagnóstico humano.

## **Despliegue y distribución** {#despliegue-y-distribución}

El modelo final será convertido a formato TensorFlow Lite (.tflite) aplicando cuantización post-entrenamiento a formato Int8, con el objetivo de reducir el tamaño del archivo y mejorar la eficiencia de ejecución en CPU móvil.  
La funcionalidad principal de la aplicación permitirá la captura de imagen mediante cámara, el procesamiento local sin conexión a internet y la visualización de clase predicha y nivel de confianza. Se implementará además un módulo de sincronización que permita:

* Almacenamiento local de imágenes capturadas con etiquetado manual.  
* Registro de fecha.  
* Registro opcional de ubicación aproximada (bajo consentimiento).  
* Envío seguro de datos a una API cuando exista conexión a internet.

La API permitirá la recepción, almacenamiento y consulta estructurada de imágenes. Este mecanismo permitirá la construcción progresiva de un repositorio de imágenes reales capturadas en territorio salvadoreño.  
La aplicación y la API estarán diseñadas para soportar:

* Consulta de versión de modelo disponible.  
* Descarga de nuevos modelos TensorFlow Lite cuando exista conectividad.  
* Reemplazo seguro del modelo local sin necesidad de reinstalar la aplicación.

De esta forma, el sistema permitirá mejoras iterativas del modelo sin afectar la funcionalidad offline principal. El reentrenamiento no se ejecutará dentro del alcance del proyecto, pero se documentará formalmente el procedimiento técnico para la curaduría y validación de nuevas imágenes recolectadas, el balanceo y limpieza del nuevo subconjunto salvadoreño y el fine-tuning incremental del modelo base utilizando las nuevas imágenes. Este protocolo permitirá que futuras investigaciones o instituciones académicas puedan ampliar el dataset nacional, mejorar la robustez del modelo y contribuir a la comunidad científica y agrícola.

# **Cronograma** {#cronograma}

# **Recursos necesarios** {#recursos-necesarios}

Para la ejecución del proyecto, se requieren recursos humanos, tecnológicos, computacionales y operativos que garanticen el adecuado desarrollo, entrenamiento, validación y despliegue del sistema propuesto.

## **Recursos Humanos** {#recursos-humanos}

* Especialistas en Machine Learning: Responsables del diseño experimental, selección de arquitectura, entrenamiento, evaluación, optimización y documentación técnica del modelo.  
* Desarrolladores Full Stack: Encargados del desarrollo de la API, la aplicación web progresiva (PWA), la integración del modelo entrenado y la implementación del módulo de sincronización.

## **Recursos Computacionales** {#recursos-computacionales}

El proyecto contempla una estrategia híbrida de entrenamiento, combinando recursos locales y servicios en la nube.

**Entorno local**  
Se utilizará un equipo de cómputo con las siguientes características recomendadas:

* Procesador multinúcleo moderno.  
* 16 GB de memoria RAM (mínimo recomendado).  
* Unidad de almacenamiento SSD.  
* GPU dedicada (preferiblemente NVIDIA compatible con CUDA) para acelerar el entrenamiento.

**Google Colab**  
Se empleará Google Colab como entorno de entrenamiento en la nube para:

* Entrenamiento principal del modelo.  
* Fine-tuning con transferencia de aprendizaje.  
* Experimentación con diferentes configuraciones.

En caso de ser requerido se contempla la posibilidad de migrar temporalmente a Google Colab Pro o Pro+, cuyo costo mensual es moderado y justificable en fases de optimización intensiva.

## **Recursos de Software** {#recursos-de-software}

**Entrenamiento y modelado**

* Python 3.x con TensorFlow, librerías de procesamiento de imágenes (OpenCV, Pillow), herramientas de data augmentation y frameworks para conversión y optimización del modelo

**Backend**

* FastAPI y Base de datos MySQL

**Frontend (PWA)**

* Vue 3

**Control de versiones y automatización**

* Git y GitHub con posible configuración de integración y despliegue continuo (CI/CD)

## **Recursos de Datos** {#recursos-de-datos}

* Dataset inicial de imágenes de enfermedades del maíz (fuentes públicas y/o académicas).  
* Imágenes reales recopiladas en campo.

## **Infraestructura de Despliegue** {#infraestructura-de-despliegue}

* Servidor en la nube (Linux).

## **Recursos Económicos** {#recursos-económicos}

* Hosting o VPS.  
* Dominio y certificado SSL.  
* Eventual suscripción temporal a Google Colab Pro (si se requiere).  
* Costos operativos asociados a recolección de imágenes en campo.  
* Consumo eléctrico del equipo local durante entrenamientos prolongados.

# **Riesgos** {#riesgos}

- Existe una alta probabilidad de que el modelo alcance una precisión casi perfecta con imágenes de laboratorio de los dataset seleccionados, pero fracase en el campo real debido a variables no controladas como la iluminación, sombras o maleza de fondo.  
- En el contexto agronómico, decirle a un agricultor que su cultivo está sano cuando en realidad tiene un inicio de una plaga (un falso negativo) es mucho más destructivo que diagnosticar erróneamente una enfermedad leve en una planta sana (falso positivo).  
- Dado que el sistema debe funcionar completamente *offline* en las parcelas, la inferencia se realizará utilizando el hardware del dispositivo móvil del agricultor, los cuales suelen ser teléfonos Android de gama baja con memoria RAM y capacidad de procesamiento limitadas. Un modelo pesado drenará la batería o colapsará la aplicación.  
- Un modelo de visión por computadora es tan bueno como la imagen que procesa. Si el agricultor toma fotografías borrosas, desde muy lejos o a contraluz, el modelo emitirá diagnósticos aleatorios.

# **Bibliografía** {#bibliografía}

- CRISP-DM Explained: A Proven Data Mining Methodology \- Udacity, fecha de acceso: abril 11, 2026, [https://www.udacity.com/blog/2025/03/crisp-dm-explained-a-proven-data-mining-methodology.html](https://www.udacity.com/blog/2025/03/crisp-dm-explained-a-proven-data-mining-methodology.html)  
- Ministerio de Agricultura y Ganadería (MAG). (2023). *Anuario de Estadísticas Agropecuarias 2022-2023*. Dirección General de Economía Agropecuaria, Gobierno de El Salvador, fecha de acceso: abril 11, de 2026, [https://www.mag.gob.sv/anuarios-de-estadisticas-agropecuarias/](https://www.mag.gob.sv/anuarios-de-estadisticas-agropecuarias/)  
- El Diario de Hoy. Recuperado de Reporte de cosecha de granos básicos 2023 **–** el diario de hoy, fecha de acceso: abril 11, de 2026, [https://www.elsalvador.com/h-noticias/h-negocios/bcr-agricultura-ganaderia-el-salvador-pib/1135356/2024/](https://www.elsalvador.com/h-noticias/h-negocios/bcr-agricultura-ganaderia-el-salvador-pib/1135356/2024/)  
- Deng, J., et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*.  
  [https://www.image-net.org/](https://www.image-net.org/)  
- TensorFlow. (2024). *TensorFlow Lite Guide*. [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)  
- Centro Nacional de Tecnología Agropecuaria y Forestal (CENTA). (2018). Cultivo de Maíz (Zea mays L.): Guía Técnica. Ministerio de Agricultura y Ganadería, El Salvador. fecha de acceso: abril 11, de 2026, [https://www.centa.gob.sv/download/guia-tecnica-cultivo-de-maiz/](https://www.centa.gob.sv/download/guia-tecnica-cultivo-de-maiz/)  
- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.  
  [https://www.manning.com/books/deep-learning-with-python-second-edition](https://www.manning.com/books/deep-learning-with-python-second-edition)  
- Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*.[https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)  
- Deng, J., et al. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*.  
  [https://www.image-net.org/](https://www.image-net.org/)  
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*.  
  [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)  
- Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation*. Journal of Big Data.  
  [https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)  
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. [https://web.stanford.edu/\~hastie/ElemStatLearn/](https://web.stanford.edu/~hastie/ElemStatLearn/)

# **Anexos** {#anexos}

## **Recopilación de datasets** {#recopilación-de-datasets}

- [https://www.kaggle.com/datasets/hamishcrazeai/maize-in-field-dataset](https://www.kaggle.com/datasets/hamishcrazeai/maize-in-field-dataset)

Fotos en entornos reales con otros detalles presentes en las fotos (manos y otros artefactos). Las etiquetas de la enfermedad están en un CSV por lo que requerirá tratado.

- [https://www.kaggle.com/datasets/kaustavbiswal/maize-diseases](https://www.kaggle.com/datasets/kaustavbiswal/maize-diseases)

Imágenes provenientes de los datasets PlantVillage y PlantDoc, además de algunas enfermedades con una actualización subida con fotos en entornos de campo.

- [https://www.kaggle.com/datasets/shuvokumarbasak2030/corn-leaf-diseases-plant-village-augmented-data](https://www.kaggle.com/datasets/shuvokumarbasak2030/corn-leaf-diseases-plant-village-augmented-data)

Recopilación de PlantVillage-AD (Augmented Dataset), es decir un dataset al que se le han aplicado técnicas de augmentation para mejorar la capacidad de generalización.

- [https://www.kaggle.com/datasets/saniyaverma914/cropdg-unified-multidomain](https://www.kaggle.com/datasets/saniyaverma914/cropdg-unified-multidomain)

Caso similar al dataset anterior, construido a partir de 3 datasets públicos, está dividido en dominios, pero la enfermedad está un poco generalizada.

- [https://www.kaggle.com/datasets/osutokaggle/maize-beans-and-tomatoes-image-dataset-for-africa](https://www.kaggle.com/datasets/osutokaggle/maize-beans-and-tomatoes-image-dataset-for-africa)

Recopilación bastante extensa (10 datasets) de fotos en entornos reales de distintas plantas, tomadas en varios países de África.
