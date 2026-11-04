# Usar model.joblib en Detectcam

La aplicación admite detección YOLO con un clasificador Joblib opcional, o
detección solo SVM mediante ventanas multiescala. También puede clasificar un
recorte o una matriz de características sin abrir la cámara ni la interfaz.

## Qué contiene el archivo proporcionado

Se comprobó la carga de `/home/simanem/Downloads/model.joblib` con estas propiedades:

| Propiedad | Valor |
| --- | --- |
| Objeto | `sklearn.pipeline.Pipeline` |
| Primer paso | `StandardScaler` |
| Clasificador | `LinearSVC(dual=False, max_iter=20000, random_state=0)` |
| Entrada | 1,812 características numéricas por muestra |
| Salida | Identificador `0`, `1`, `2`, `3` o `4` |
| Versión de scikit-learn registrada | `1.9.1` |

El script proporcionado `train_select_svm_rf_valid_only_v1.py` define este mapa,
guardado en `model/classes.json`:

| Identificador | Clase |
| --- | --- |
| 0 | `drone` (dron) |
| 1 | `airplane` (avión) |
| 2 | `helicopter` (helicóptero) |
| 3 | `bird` (ave) |
| 4 | `background` (fondo) |

`StandardScaler` resta la media y divide por la escala aprendidas para cada
característica. Después, `LinearSVC` calcula puntuaciones con sus coeficientes
y selecciona una clase. Los dos pasos ya están entrenados: no hay que volver a
normalizar con otro scaler ni llamar a `fit` al ejecutar la aplicación.

El modelo no incluye el extractor ni entrega coordenadas de objetos. La
transformación original se recuperó de `extract_hog_hsv_features_v1.py` y su
`feature_config.json`; está adaptada para inferencia en `ptz/hog_hsv_features.py`.

El script de entrenamiento lee una caché llamada `hog_hsv_features_v1`, con
`feature_config.json`, `feature_index.csv` y `features.float32.npy`. La inferencia
en vivo no necesita cargar esa matriz ni instalar pandas.

## Cómo se convierte cada recorte en características

1. Se recibe la imagen BGR de OpenCV y se conserva su proporción, ajustando su
   lado mayor a 64 píxeles. Al reducir se usa `INTER_AREA`; al ampliar,
   `INTER_CUBIC`.
2. Para HOG se completa un canvas de 64 × 64 mediante reflejo
   `BORDER_REFLECT_101` (o `BORDER_REPLICATE` si alguna dimensión tiene un píxel)
   y se convierte a gris. HOG usa 9 orientaciones, celdas de 8 × 8, bloques de
   2 × 2 y normalización `L2-Hys`: produce **1,764 valores**.
3. Sobre el contenido redimensionado, antes del relleno, se calculan histogramas
   separados H, S y V de 16 bins cada uno. Cada histograma suma 1: son
   **48 valores** adicionales.
4. Se concatenan HOG, H, S y V, en ese orden, en un vector `float32` de
   **1,812 valores**. Ese vector entra al pipeline ya entrenado.

El extractor usa scikit-image `0.26.0`, la versión registrada en la caché.

Durante el entrenamiento se comparan SVM lineal y Random Forest con los perfiles
`bg1x` y `bg2x`. Para cada familia se selecciona el perfil por F1 macro en
validación; los desempates usan F1 de dron, exactitud balanceada y nombre del
perfil. El script no evalúa TEST. El `.joblib` proporcionado corresponde a un
SVM; sin los CSV de resultados no se puede determinar su perfil seleccionado
ni su precisión real.

Las puntuaciones de `decision_function` son márgenes, **no probabilidades**.
Consulte las referencias oficiales de
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
y [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

## Inspeccionar y probar sin cámara

Desde la carpeta que contiene `main.py`:

```bash
source .venv/bin/activate
python -m pip install -r requirements-joblib.txt
python -m ptz.joblib_classifier --model model/model.joblib --labels model/classes.json
```

El modelo proporcionado se copió a `model/model.joblib`, ignorado por Git.
Puede indicar otra ruta con `--model`. El comando muestra el número de
características, los pasos y los identificadores de las clases.

Si ya dispone de características calculadas con la función de entrenamiento,
guárdelas en un `.npy` con forma `(N, 1812)` y ejecute:

```bash
python -m ptz.joblib_classifier --model model/model.joblib --labels model/classes.json --features caracteristicas.npy
```

También admite un vector `(1812,)` para una muestra. Se rechazan dimensiones
incorrectas, valores no numéricos, NaN e infinitos. El resultado incluye la
clase y su margen SVM. Este comando recibe características, no imágenes.

Para clasificar una imagen de un objeto o un recorte con el extractor integrado:

```bash
python -m ptz.joblib_classifier --labels model/classes.json --image recorte.jpg
```

Este comando clasifica toda la imagen recibida como una muestra; no localiza
varios objetos dentro de ella. Puede indicar otro archivo extractor con
`--extractor ruta/features.py`. `--image` y `--features` son alternativas.

## YOLO con clasificación SVM en video

El flujo implementado es:

```text
Video RTSP → YOLO → cajas → recortes BGR → extractor original
           → StandardScaler → LinearSVC → clase → seguimiento/PTZ
```

YOLO sigue siendo necesario para localizar objetos. Si YOLO no encuentra un
objeto, el clasificador tampoco lo recibirá. Esta combinación necesita validarse
con recortes representativos de las imágenes usadas durante el entrenamiento.

La configuración local `.env` ya quedó activada con estas tres rutas. En otra
instalación, después de instalar `requirements-joblib.txt`, configure:

```dotenv
# Conserve aquí sus pesos o su exportación NCNN actual.
DETECTCAM_DETECTOR=yolo
DETECTCAM_MODEL_PATH=model/best.pt
DETECTCAM_CLASSIFIER_PATH=model/model.joblib
DETECTCAM_CLASSIFIER_FEATURES_PATH=ptz/hog_hsv_features.py
DETECTCAM_CLASSIFIER_LABELS_PATH=model/classes.json
```

La línea de `DETECTCAM_MODEL_PATH` es ilustrativa: si ya utiliza una exportación
NCNN, conserve su ruta actual. Las rutas relativas se resuelven desde la carpeta
de `main.py`. Ajuste `DETECTCAM_TARGET_CLASSES` a los nombres reales del JSON
que quiera seguir y ejecute:

```bash
python main.py
```

La pestaña Seguimiento muestra las cuatro clases de objetos. Las predicciones
`background` se descartan antes del seguimiento. Las etiquetas del
video separan `YOLO:0.85` (confianza de la detección original) de `SVM:+1.20`
(margen de la nueva clasificación). El umbral de confianza de seguimiento sigue
filtrando la confianza YOLO; no mide la certeza de la clase SVM. El clasificador
siempre asigna una de sus cinco clases; la clase `background` permite rechazar
recortes clasificados como fondo. Esto no garantiza detectar cualquier objeto
desconocido como fondo.
La telemetría de inferencia incluye detección, extracción y clasificación.
LinearSVC se ejecuta en CPU; `DETECTCAM_DEVICE` sigue seleccionando el dispositivo
de YOLO.

En esta Steam Deck, desde el 4 de noviembre de 2026, la configuración local usa
`DETECTCAM_MODEL_PATH=model/best_416_ncnn_model`, `DETECTCAM_DEVICE=vulkan:0` y
`DETECTCAM_IMGSZ=416`: YOLO se ejecuta con NCNN/Vulkan en la GPU AMD y el
clasificador continúa en CPU. Se comprobó el arranque con recepción de video.

Para regresar a YOLO solamente, deje vacías las tres variables
`DETECTCAM_CLASSIFIER_*`. Se conservó una copia de la configuración anterior en
`.env.before-joblib-20261103`.

## Detección solo con SVM, sin cargar YOLO

El modo `svm` recorre la imagen con ventanas cuadradas superpuestas a varios
tamaños. Aplica el mismo extractor HOG/HSV y el SVM a todas las ventanas en un
lote, descarta fondo y puntuaciones bajas, y elimina cajas repetidas con NMS
por clase. Las coordenadas se convierten al tamaño original para el seguimiento.

El `LinearSVC` original no produce probabilidades. Para que el filtro de
confianza del seguimiento use probabilidades estimadas mediante calibración,
se prepara una copia con `CalibratedClassifierCV(FrozenEstimator(model))`:

```bash
.venv/bin/python -m ptz.calibrate_svm
```

El comando carga `model/model.joblib` y selecciona exclusivamente las **41,931
muestras `valid_natural`** de la caché local. Ajusta calibradores sigmoid con
sus etiquetas, conservando los pesos del SVM y del scaler. No selecciona filas
de TRAIN ni TEST. Produce `model/model_calibrated.joblib` y un informe JSON con
hashes, clases y cantidad de muestras. La copia calibrada ya está preparada en
esta máquina. Para repetir el cálculo sobre el mismo destino use `--overwrite`.

La validación natural ya se usó para seleccionar el perfil del modelo original.
Esta calibración no constituye una evaluación independiente de precisión, y
las ventanas sobre video pueden tener otra distribución que los recortes de
validación. La calibración también puede cambiar la clase elegida respecto del
SVM original. Hay que comprobar sus resultados con video representativo.

Para activar este modo en `.env` y reiniciar la aplicación:

```dotenv
DETECTCAM_DETECTOR=svm
DETECTCAM_CLASSIFIER_PATH=model/model_calibrated.joblib
DETECTCAM_CLASSIFIER_FEATURES_PATH=ptz/hog_hsv_features.py
DETECTCAM_CLASSIFIER_LABELS_PATH=model/classes.json
DETECTCAM_SVM_SCAN_WIDTH=320
DETECTCAM_SVM_WINDOW_SIZES=32,64,128
DETECTCAM_CONFIDENCE=0.50
```

```bash
./run.sh
```

`DETECTCAM_MODEL_PATH`, `DETECTCAM_DEVICE` y `DETECTCAM_IMGSZ` no seleccionan
modelos ni GPU en modo SVM: no se carga YOLO. Esta implementación se ejecuta
en **CPU**. El ancho de búsqueda limita el trabajo; a 320 y con imagen 16:9,
los tres tamaños predeterminados producen aproximadamente 230 ventanas por
cuadro. Un ancho mayor permite buscar objetos más pequeños, pero cuesta más
tiempo. Los tamaños cuadrados y los pasos de barrido limitan las formas y
escalas que se pueden localizar. Las cajas representan ventanas candidatas,
no una regresión aprendida de los bordes del objeto.

El umbral de detección y el de seguimiento filtran probabilidades calibradas;
el modelo sin calibrar se rechaza en este modo con un mensaje accionable. El
flujo YOLO + SVM anterior permanece disponible con `DETECTCAM_DETECTOR=yolo`
y `DETECTCAM_CLASSIFIER_PATH=model/model.joblib`. La configuración activa de
esta Steam Deck se conserva en ese flujo con GPU mientras se añade esta opción.

Los pesos originales y calibrados y la matriz grande de características se
mantienen como archivos locales ignorados por Git. El repositorio contiene el
código, las etiquetas y la configuración de características para reproducirlo;
en otra instalación hay que proporcionar los modelos y, para calibrar, la caché.

## Estado de la validación

Se verificaron la carga del archivo real y la equivalencia de las predicciones y
márgenes con el pipeline original. Las pruebas automatizadas comparan el nuevo
extractor con las funciones originales usando imágenes de tamaños y proporciones
distintos, incluidas dimensiones de un píxel. También verifican los histogramas
sin relleno, el orden, los recortes, las clases y la configuración.

La caché disponible contiene 85,140 muestras de 1,812 características, pero no
incluye los recortes originales. Por eso no se pudo comparar una imagen del
dataset con su fila almacenada. Estas comprobaciones validan la integración;
no miden precisión sobre video ni validan seguimiento con la cámara.

Use las versiones del entrenamiento para cargar modelos persistidos. La versión
`1.9.1` se obtuvo del propio archivo. La caché registra NumPy `2.5.3`, OpenCV
`5.0.0` y scikit-image `0.26.0`. En este entorno se probó con NumPy `2.5.2`,
OpenCV `5.0.0`, scikit-image `0.26.0`, SciPy `1.18.1` y Joblib `1.6.0`; las
versiones originales de SciPy y Joblib no están registradas en estos archivos. Véase la
[documentación de persistencia de scikit-learn](https://scikit-learn.org/stable/model_persistence.html).
