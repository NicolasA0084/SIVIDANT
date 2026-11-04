# Steam Deck: instalación y aceleración AMD (Ubuntu 24.04)

Esta máquina tiene una APU AMD Van Gogh y Python 3.12. CUDA requiere una GPU
NVIDIA. La GPU AMD se puede utilizar mediante **NCNN con Vulkan**, conservando
PyTorch y Torchvision en sus versiones para CPU. No hace falta instalar CUDA ni
ROCm para esta ruta.

Todos los comandos siguientes se ejecutan desde el directorio que contiene
`main.py`, `run.sh` y `.env`.

## Ejecutar

```bash
./run.sh
```

El lanzador utiliza automáticamente `.venv/bin/python`. En el editor, seleccione
ese mismo intérprete. Se necesita una sesión de escritorio para abrir la interfaz.

Antes de iniciar, complete en `.env`:

- `DETECTCAM_CAMERA_IP`: IP o nombre de la cámara.
- `DETECTCAM_CAMERA_USER` y `DETECTCAM_CAMERA_PASSWORD`: credenciales.
- `DETECTCAM_MODEL_PATH`: archivo de pesos o directorio NCNN del modelo.
- `DETECTCAM_TARGET_CLASSES`: nombres de las clases aprendidas por el modelo.

Los pesos entrenados se proporcionan por separado; conserve `model/best.pt`
después de convertirlo. El control físico de la cámara depende de
`DETECTCAM_PTZ_HARDWARE_ENABLED`; la aplicación siempre inicia desarmada.

## Usar la GPU AMD con NCNN y Vulkan

Después de instalar las dependencias base, instale las opcionales:

```bash
.venv/bin/python -m pip install -r requirements-ncnn.txt
```

El paquete NCNN debe disponer de soporte Vulkan y el sistema debe detectar la
GPU con su controlador Vulkan. En esta instalación, `vulkan:0` corresponde a
la GPU AMD con RADV VANGOGH. El otro dispositivo detectado, `llvmpipe`, ejecuta
por software en CPU y no sirve para medir aceleración de la GPU AMD.

Para exportar los pesos a una entrada de 416 × 416:

```bash
.venv/bin/python - <<'PY'
from ultralytics import YOLO

YOLO('model/best.pt').export(
    format='ncnn', imgsz=416, device='cpu', quantize=32,
)
PY
```

La conversión se realiza en CPU y produce `model/best_ncnn_model`. Configure
estas variables en `.env` para usar esa exportación:

```dotenv
DETECTCAM_MODEL_PATH=model/best_ncnn_model
DETECTCAM_DEVICE=vulkan:0
DETECTCAM_IMGSZ=416
```

La ruta apunta al **directorio completo**, que debe conservar su archivo
`.param`, su `.bin` homónimo y `metadata.yaml`. No copie únicamente el `.bin`.
El directorio debe mantener un nombre terminado en `_ncnn_model` para que
Ultralytics identifique el formato.

También se prepararon dos exportaciones con nombres separados:

| Directorio del modelo | `DETECTCAM_IMGSZ` | Entrada del modelo |
| --- | ---: | --- |
| `model/best_416_ncnn_model` | `416` | 416 × 416 |
| `model/best_640_ncnn_model` | `640` | 640 × 640 |

Para usar una, ponga ese directorio en `DETECTCAM_MODEL_PATH` y el tamaño
correspondiente en `DETECTCAM_IMGSZ`, manteniendo `DETECTCAM_DEVICE=vulkan:0`.
Reinicie la aplicación después de cambiar `.env`.

En la configuración local se activó `model/best_416_ncnn_model` con
`vulkan:0` y `416`. La copia privada de la configuración anterior está en
`.env.before-ncnn-20261023` y está excluida de Git.

Estas exportaciones tienen tamaños **estáticos**: sus metadatos fijan la
dimensión exportada. Cambiar solo `DETECTCAM_IMGSZ` no convierte un modelo de
640 a 416; hay que seleccionar la otra exportación o volver a exportar.
La entrada de 416 reduce el trabajo, pero puede perder objetos pequeños.

### Configuración con el clasificador SVM (4 de noviembre de 2026)

La configuración local quedó en `model/best_416_ncnn_model`, `vulkan:0` y `416`,
manteniendo el clasificador `model/model.joblib`, el extractor
`ptz/hog_hsv_features.py` y las etiquetas `model/classes.json`. La red YOLO usa
la GPU AMD; HOG, HSV, StandardScaler y LinearSVC siguen en CPU.

Se verificó el backend NCNN con `net.opt.use_vulkan_compute=True` y la GPU 0
`AMD Custom GPU 0405 (RADV VANGOGH)`. Se abrió la interfaz en el escritorio y
el registro confirmó recepción de video de la cámara. Para volver a arrancar,
ejecute `./run.sh`. La configuración anterior se conservó en
`.env.before-steamdeck-gpu-20261104`.

El registro del arranque está en
`runs/detectcam-steamdeck-gpu-20261104.log`. La comprobación GPU independiente
está en `runs/steamdeck-gpu-check-20261104.json`; sus tiempos corresponden a un
cuadro sintético sin objetos, no a los FPS del video con clasificación.

El primer arranque puede tardar más mientras carga el modelo y compila shaders
Vulkan. Evalúe los FPS después del calentamiento. El preprocesamiento y NMS
siguen consumiendo CPU, aunque NCNN ejecute el modelo con Vulkan.

### Rendimiento medido el 23 de octubre de 2026

Se ejecutó el modelo entrenado `best.pt` (YOLO26m, cuatro clases) y sus
exportaciones NCNN en esta Steam Deck, sin otra instancia de Detectcam abierta.
Se utilizó un cuadro sintético de 1280 × 720, dos llamadas de calentamiento
y la mediana de nueve llamadas completas a `YOLODetector.detect()`.

| Tamaño configurado | PyTorch CPU | NCNN CPU | NCNN GPU AMD/Vulkan |
| --- | ---: | ---: | ---: |
| 640 | 3.62 FPS / 276.13 ms | 2.01 FPS / 496.55 ms | 7.38 FPS / 135.47 ms |
| 416 | 7.10 FPS / 140.89 ms | 4.67 FPS / 214.32 ms | 14.61 FPS / 68.46 ms |

La configuración GPU resultó aproximadamente dos veces más rápida que PyTorch
CPU al mismo tamaño configurado. La fila de NCNN CPU muestra que instalar o
convertir el modelo por sí solo no ofrece esa mejora: hay que usar `vulkan:0`.
Estos FPS son del detector, sin recepción RTSP, pantalla ni control PTZ; no son
una garantía de rendimiento sostenido con la cámara. La imagen sintética no
permite medir precisión. Las formas de entrada y cabezas utilizadas difieren
como se explica a continuación.

Los resultados completos, muestras y script de medición se conservan en
`.venv/ncnn-exports/benchmark.json` y
`.venv/ncnn-exports/benchmark_ncnn.py`. La GPU verificada es
`AMD Custom GPU 0405 (RADV VANGOGH)`, índice Vulkan 0.

La instalación y la selección `vulkan:0` siguen la
[guía oficial de NCNN en Ultralytics](https://docs.ultralytics.com/integrations/ncnn).

### Qué cambia al exportar YOLO26

Ultralytics avisa que NCNN no admite la rama `end2end` y la desactiva durante la
exportación. Conserva las clases y utiliza la cabeza `one2many` con NMS para
eliminar cajas superpuestas. El `.pt` habitual utiliza la cabeza `one2one` sin
NMS, por lo que las cajas y confianzas pueden variar entre ambas versiones.

Un benchmark con imágenes sintéticas comprueba velocidad y ejecución; no
demuestra precisión sobre drones ni sustituye una prueba con imágenes reales
representativas. Para comparar la conversión numéricamente, use una instancia
recién cargada del `.pt` con `end2end=False` y la misma entrada cuadrada
(`rect=False`), tamaño y umbrales que NCNN.

Se realizó esa comparación a 416 con dos imágenes de ejemplo incluidas en
Ultralytics y una imagen de ruido con semilla fija. Las clases, dimensiones
de salida, valores finitos y rango de probabilidades coincidieron con lo
esperado. NCNN CPU reprodujo las probabilidades con error absoluto máximo
menor de 0.000001. Vulkan utiliza cálculos de menor precisión: en esas muestras
el error máximo de probabilidad fue 0.00147 y hubo diferencias en las
coordenadas de propuestas antes de filtrar. Esto comprueba la conversión
numérica, pero no demuestra equivalencia de detecciones ni precisión en campo.
Los detalles están en `.venv/ncnn-exports/output-validation.json`.

Al comparar rendimiento, tenga en cuenta que PyTorch suele usar relleno
rectangular: un cuadro de 1280 × 720 a `imgsz=640` entra como 384 × 640,
mientras esta exportación NCNN utiliza 640 × 640. A `imgsz=416`, las entradas
son 256 × 416 y 416 × 416, respectivamente.

## Volver al modelo PyTorch en CPU

Cambie **también la ruta del modelo**, no solo el dispositivo:

```dotenv
DETECTCAM_MODEL_PATH=model/best.pt
DETECTCAM_DEVICE=cpu
DETECTCAM_IMGSZ=416
```

Después reinicie con `./run.sh`. Para priorizar objetos pequeños puede probar
`640`; `320` reduce el trabajo a costa de resolución. Poner `cpu` con una ruta
NCNN ejecuta NCNN en CPU y no vuelve al modelo `.pt` original.

## Entorno instalado

Los paquetes Python viven en `.venv`; no se modificó el entorno Python del
sistema. La pareja instalada es PyTorch `2.14.1+cpu` / Torchvision
`0.29.1+cpu`. La instantánea de la instalación base está en
`.venv/installed-requirements.txt`.

PyTorch y Torchvision se instalaron desde el índice oficial de ruedas CPU antes
de instalar los requisitos base:

```bash
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

OpenGL y GLib ya estaban disponibles. Como instalar paquetes del sistema
requiere una contraseña de sudo, se descargaron los paquetes Ubuntu
`python3-tk`, `libtk8.6`, `blt` y `tk8.6-blt2.5`, y se extrajeron en
`.venv/ubuntu-tk`. Un archivo privado `detectcam_local_tk.pth` y un módulo
auxiliar cargan esas bibliotecas cuando falta Tkinter del sistema. Esta copia
es específica de Ubuntu 24.04, x86_64 y Python 3.12.

Si recrea `.venv`, puede instalar Tkinter normalmente desde su terminal:

```bash
sudo apt install python3-venv python3-tk libgl1
```

Después siga la instalación CPU del README y añada `requirements-ncnn.txt`
para disponer de NCNN/Vulkan.

## Verificar sin cámara

```bash
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
```

Estas pruebas utilizan imágenes sintéticas y control de cámara simulado.
No validan la conexión RTSP/PTZ real ni la precisión del modelo entrenado.

El 23 de octubre de 2026 pasaron `pip check` y las 95 pruebas actuales,
incluidas las nuevas comprobaciones de exportaciones NCNN completas e
incompletas y las pruebas de interfaz. El registro está en
`.venv/ncnn-test-results.log`; las versiones instaladas con los extras se
guardaron en `.venv/installed-requirements-ncnn.txt`.

### Validación histórica del 19 de octubre de 2026

En la instalación inicial pasaron `pip check` y las 90 pruebas disponibles,
incluidas las pruebas gráficas a 1280 × 800. También pasaron las importaciones,
una operación de tensor en CPU, NMS de Torchvision e inferencia real en CPU a
través de `YOLODetector`.

Aquella inferencia utilizó pesos temporales **sin entrenar** de YOLO11n y una
imagen sintética de 320 × 320. En esa fecha todavía no se habían probado los
pesos de la USB ni el hardware RTSP/PTZ. El registro de esa revisión permanece
en `.venv/test-results.log`; no representa la validación posterior de NCNN.
