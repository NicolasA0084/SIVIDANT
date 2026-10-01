# Detectcam

Detectcam recibe video RTSP, ejecuta detección YOLO, selecciona y sigue un
objetivo y calcula decisiones de seguimiento PTZ. Puede controlar cámaras
Hikvision mediante ISAPI con ONVIF como alternativa de conexión.

El arranque predeterminado es de **simulación: no conecta el backend PTZ ni
habilita movimiento físico**. Habilitar hardware tampoco arma el ejecutor; el
armado requiere una acción y confirmación manual independientes.

## Funcionalidad presente en el código

- adquisición RTSP en segundo plano y reconexión;
- inferencia con un modelo local de Ultralytics YOLO;
- selección y tracking temporal de un objetivo;
- cálculo y visualización de decisiones PTZ;
- interfaz con video limpio, registro plegable y control manual/automático;
- backend ISAPI y fallback ONVIF durante la conexión inicial;
- interlocks, desarmado, parada de emergencia, dead-man y watchdog;
- métricas opcionales en `runs/`.

No están habilitados en el flujo físico Position3D, autozoom, búsqueda automática
ni Smart Tracking.

## Requisitos

- Linux de escritorio (Ubuntu/Debian) y Python 3.12;
- `python3-venv`, `python3-tk` y `libgl1`;
- escritorio gráfico para la interfaz Tkinter;
- acceso RTSP a la cámara;
- ISAPI u ONVIF solamente si se decide habilitar el control físico;
- pesos de detección YOLO compatibles con las clases configuradas;
- opcional: GPU NVIDIA con driver compatible para acelerar la inferencia.

Las versiones de `requirements.txt` son las dependencias directas comprobadas en
el entorno Python 3.12 usado para esta revisión. PyTorch se instala por separado
para elegir CPU o la variante CUDA apropiada.

## Instalación inicial

Ejecute los comandos desde la carpeta que contiene `main.py` y `requirements.txt`.
Los pasos de sistema corresponden a Linux; el bloque de drivers usa las
herramientas de Ubuntu. Use Python 3.12 para reproducir el entorno comprobado.

### 1. Preparar Python y el entorno virtual

```bash
sudo apt update
sudo apt install git python3-full python3-venv python3-tk libgl1
python3 --version
```

Si muestra Python 3.12, cree el entorno:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Si su distribución usa otra versión, instale Python 3.12 con sus módulos `venv`
y Tkinter mediante los paquetes disponibles para esa distribución, y cree el
entorno con `python3.12 -m venv .venv`.

### 2. Instalar PyTorch: elegir CPU o GPU NVIDIA

Instale **una sola opción** en el entorno virtual activo. PyTorch y Torchvision
se instalan juntos desde el mismo índice para mantener una combinación compatible.

#### Opción A: CPU

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Mantenga `DETECTCAM_DEVICE=cpu` en `.env`.

#### Opción B: GPU NVIDIA en Ubuntu

Compruebe primero que el sistema reconoce la tarjeta:

```bash
nvidia-smi
```

Debe mostrar el modelo de GPU y la versión del driver. Si el driver aún no está
instalado, use las herramientas de Ubuntu:

```bash
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers list
sudo ubuntu-drivers install
```

Guarde su trabajo y reinicie el equipo después de instalar el driver. Si el
instalador solicita registrar una clave MOK para Secure Boot, complete ese paso
durante el reinicio. Consulte la [guía oficial de drivers de Ubuntu](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/)
para problemas de módulos o firma; estos comandos de `ubuntu-drivers` son
específicos de Ubuntu.

Después del reinicio, vuelva a la carpeta del proyecto:

```bash
nvidia-smi
source .venv/bin/activate
```

Elija la variante de PyTorch según **el driver y la arquitectura de su GPU**.
Por ejemplo, para un equipo compatible con CUDA 12.6:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Para una GPU compatible con CUDA 13.0 y un driver de la serie 580 o posterior,
la alternativa es:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Son alternativas, no pasos consecutivos. Consulte las [combinaciones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/)
y la [compatibilidad de drivers de NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
si necesita fijar versiones. Las tarjetas antiguas pueden requerir una versión
anterior de PyTorch/CUDA; las recientes necesitan una compilación que incluya
su arquitectura. El [selector de PyTorch](https://pytorch.org/get-started/locally/)
permite consultar las opciones de instalación vigentes.

La instalación binaria de PyTorch obtiene las bibliotecas CUDA necesarias; para
ejecutar este proyecto no necesita instalar por separado CUDA Toolkit, cuDNN,
`nvcc` ni configurar `CUDA_HOME`. El driver NVIDIA sí debe estar instalado en
el sistema. Véase la [explicación del equipo de PyTorch](https://discuss.pytorch.org/t/do-i-need-to-uninstall-cuda-11-6/150568).
El campo `CUDA Version` de `nvidia-smi` indica la compatibilidad del driver;
`torch.version.cuda` identifica la variante de PyTorch instalada. Consulte la
[referencia de NVIDIA](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html).

Si ya instaló la opción CPU y quiere cambiar a GPU, ejecute dentro de `.venv`:

```bash
python -m pip uninstall -y torch torchvision
# Elija el índice compatible con su equipo; aquí se usa CUDA 12.6.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. Instalar las dependencias y crear la configuración

Después de instalar PyTorch, tanto para CPU como para NVIDIA:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

En una instalación nueva, cree `.env` a partir de la plantilla:

```bash
cp .env.example .env
```

Si ya tiene un `.env`, conserve sus datos e incorpore solo los ajustes nuevos
de `.env.example`.

Para NVIDIA, edite esta línea de `.env`:

```dotenv
DETECTCAM_DEVICE=0
```

`0` selecciona la primera GPU visible para PyTorch; `1`, la segunda. La
plantilla usa `cpu`, por lo que instalar PyTorch con CUDA no activa por sí solo
la inferencia en GPU. Este ajuste acelera YOLO; la recepción y decodificación
RTSP siguen usando OpenCV con la configuración de video del proyecto.

### 4. Comprobar la instalación

Compruebe las importaciones y las versiones sin conectar la cámara:

```bash
python -c "import cv2, tkinter, torch, torchvision, ultralytics; from PIL import ImageTk; from onvif import ONVIFCamera; print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__, 'YOLO:', ultralytics.__version__)"
```

Para NVIDIA, compruebe también una operación real en la GPU seleccionada
(cambie `gpu = 0` si usará otra):

```bash
python - <<'PY'
import torch

gpu = 0
print('PyTorch:', torch.__version__)
print('CUDA de PyTorch:', torch.version.cuda)
print('CUDA disponible:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Revise el driver y la instalación CUDA de PyTorch'
print('GPU:', torch.cuda.get_device_name(gpu))
x = torch.ones((16, 16), device=f'cuda:{gpu}')
print('Operación en GPU:', (x @ x).sum().item())
torch.cuda.synchronize(gpu)
PY
```

Debe mostrar `CUDA disponible: True`, el nombre de la tarjeta y completar la
operación. Si falla, revise [Resolución de errores](#resolución-de-errores).
Esta prueba valida PyTorch en GPU; el modelo YOLO debe comprobarse después con
sus pesos y video reales.

### 5. Colocar los pesos del modelo

Los pesos no forman parte de la entrega pública: su origen y permisos de
redistribución no están documentados. Obtenga el archivo del responsable del
modelo o entrene/exporte uno propio con Ultralytics, confirme que tiene permiso
para usarlo y colóquelo, por ejemplo, en `model/best.pt`. Cree primero la carpeta:

```bash
mkdir -p model
```

También puede usar otra ubicación mediante `DETECTCAM_MODEL_PATH`. Las rutas
relativas se resuelven desde la raíz del proyecto, incluso si VS Code ejecuta
`main.py` con otro directorio de trabajo. Git ignora formatos comunes de pesos.

## Configuración

Edite `.env`; se carga automáticamente al ejecutar el programa. No es necesario
usar `source .env`, `set -a` ni exportar variables en cada sesión. Una variable
ya definida en el entorno tiene prioridad sobre el valor del archivo.

Complete primero los datos de conexión y del modelo:

| Variable | Qué configurar |
| --- | --- |
| `DETECTCAM_CAMERA_IP` | IP o host de la cámara, sin protocolo ni ruta. |
| `DETECTCAM_CAMERA_USER` | Usuario con acceso al video y, si se habilita, control PTZ. |
| `DETECTCAM_CAMERA_PASSWORD` | Contraseña de ese usuario. |
| `DETECTCAM_MODEL_PATH` | Archivo local de pesos de detección, por ejemplo `model/best.pt`. |
| `DETECTCAM_DEVICE` | `cpu` o índice de GPU NVIDIA, por ejemplo `0`. |
| `DETECTCAM_RTSP_CHANNEL` | Canal de video de la cámara; el valor inicial es `101`. |
| `DETECTCAM_TARGET_CLASSES` | Nombres de clases presentes en sus pesos, separados por comas. |
| `DETECTCAM_IMGSZ` | Tamaño de inferencia; empieza en `640`. |

Los nombres `drone,airplane,helicopter,bird` de la plantilla son ejemplos:
ajústelos a su modelo. Puede consultar las clases cargadas en la pestaña
**Seguimiento**. El filtro no añade clases que los pesos no hayan aprendido.

`.env.example` documenta los demás campos. No coloque una URL RTSP completa con
credenciales en ninguna variable. `.env` está ignorado por Git.

El valor seguro predeterminado es:

```dotenv
DETECTCAM_PTZ_HARDWARE_ENABLED=false
```

Con ese valor se procesan video, detecciones, tracking y decisiones simuladas,
pero no se crea ni conecta un backend PTZ. El armado manual continúa disponible
solo después de habilitar explícitamente el hardware; el arranque no lo automatiza.

### Preparar la cámara

1. Configure la IP de la cámara y compruebe que es accesible desde el equipo.
2. Habilite RTSP y verifique el canal de video. La aplicación usa el puerto
   RTSP `554`; no hay una variable `.env` para cambiar ese puerto actualmente.
3. Para control físico, habilite ISAPI u ONVIF en la cámara y dé al usuario
   configurado permisos PTZ. Ajuste `DETECTCAM_ISAPI_PORT`,
   `DETECTCAM_ONVIF_PORT` y `DETECTCAM_PTZ_CHANNEL` si difieren de la plantilla.
   ONVIF puede requerir un usuario creado específicamente en la cámara.
4. Empiece con `DETECTCAM_PTZ_HARDWARE_ENABLED=false` y compruebe video,
   detecciones y clases. Este modo simula el control PTZ, pero requiere una
   cámara RTSP real y pesos del modelo.
5. Para probar movimiento físico, cambie el valor a `true`, reinicie el programa,
   elija el modo y use **Armar PTZ** con su confirmación manual.

`DETECTCAM_PTZ_BACKEND=auto` intenta ISAPI y usa ONVIF si falla la conexión
inicial. También puede seleccionar `isapi` u `onvif` explícitamente.

## Ejecución habitual

Active el entorno virtual y ejecute desde cualquier directorio:

```bash
python /ruta/a/Detectcam/main.py
```

Desde la raíz:

```bash
source .venv/bin/activate
python main.py
```

El arranque valida primero `.env`, campos, rangos y el archivo de pesos. Los
errores indican la variable o archivo que debe corregirse y el logging redacta
usuario, contraseña y credenciales embebidas en URLs.

## Interfaz y controles

El mismo comando `python main.py` abre el centro de control:

- **Video**: conserva cajas de detección, objetivo y zona central; los datos ya
  no cubren la imagen. Se ajusta al tamaño de la ventana sin deformarse.
- **Registro ▾ / ▸**: muestra u oculta el panel inferior. La pestaña **Estado
  actual** muestra telemetría de video y PTZ; **Eventos** conserva hasta 2 000
  líneas, con opciones para pausar y limpiar. Al pausar, el búfer también está
  limitado a las 2 000 entradas más recientes.
- **Armar PTZ**: abre un popup «¿Seguro que quieres armar el PTZ?». Confirma
  con **Sí**; **No** cancela. **Desarmar**
  detiene y bloquea el movimiento. En simulación, estos botones están
  deshabilitados; para usar hardware debe habilitarse previamente
  `DETECTCAM_PTZ_HARDWARE_ENABLED=true` en `.env`.
- **Manual** (modo inicial): la cruceta mueve pan/tilt mientras se mantiene
  pulsado. Al soltar, salir del botón, perder foco o minimizar, se detiene.
  El deslizador permite entre 5 % y 75 % de velocidad manual de pan/tilt.
- **Seguimiento automático**: aplica las decisiones del tracker. Cambiar entre
  modos detiene y desarma; hay que volver a armar en el modo elegido.
- **Pestaña Seguimiento**: ajusta la confianza mínima de movimiento y selecciona
  una o varias clases del modelo. Cada clic marca o desmarca una clase; también
  puedes usar **Todos** y **Ninguno**. Pulsa **Aplicar** para activar los cambios.
  Sin clases seleccionadas se pausa el seguimiento automático. Si hay varias,
  la cámara sigue un objetivo a la vez entre esas clases. Los ajustes duran
  esta sesión; al reiniciar se recupera la configuración inicial.
- **Parada de emergencia** o **Espacio**: detiene y desarma. El botón permanece
  visible aunque la ventana sea pequeña; el resto de controles tiene scroll.
- Atajos: `a` abre el armado, `d` desarma y `q` cierra.

La interfaz, la visión y el envío de órdenes se ejecutan por separado. La
cruceta necesita un cuadro mostrado reciente y pulsaciones renovadas; no
continúa moviendo si la interfaz deja de responder. La inferencia lenta no
bloquea los botones. Al cerrar se detienen el PTZ, el stream y las métricas.

Para actualizar una instalación existente:

```bash
sudo apt install python3-tk
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

## Resolución de errores

| Síntoma | Qué revisar |
| --- | --- |
| No se encontró `.env` | Cree el archivo con `cp .env.example .env` y complete los campos. |
| Faltan campos o hay un valor inválido | Revise las variables indicadas, booleanos (`true`/`false`), puertos y perfil `SAFE`. |
| Modelo inexistente | Copie los pesos y corrija `DETECTCAM_MODEL_PATH`. |
| No abre la ventana | Use una sesión gráfica con Tkinter y Pillow. Compruebe `python -m tkinter`; en SSH sin pantalla la interfaz no puede abrirse. |
| Error de `libGL.so.1` | Instale `libgl1` y use el entorno virtual del proyecto. |
| RTSP no conecta | Revise IP, credenciales, canal, puerto `554` y acceso de red a la cámara. |
| `nvidia-smi` no funciona | Revise el driver del sistema y reinicie tras instalarlo; con Secure Boot, compruebe la firma del módulo. |
| `torch.version.cuda` muestra `None` | Tiene PyTorch para CPU; siga el bloque de cambio a GPU. |
| `torch.cuda.is_available()` es `False` | Compruebe el driver, la variante CUDA instalada, el intérprete activo y que `CUDA_VISIBLE_DEVICES` no oculte la tarjeta. |
| CUDA funciona, pero YOLO usa CPU | Cambie a `DETECTCAM_DEVICE=0` en `.env` y reinicie; una variable exportada con el mismo nombre tiene prioridad. |
| `no kernel image is available` o GPU no compatible | La compilación de PyTorch no incluye su arquitectura; elija una combinación Torch/Torchvision/CUDA que sí la soporte. |
| `CUDA out of memory` | Cierre otros procesos que consuman VRAM o reduzca `DETECTCAM_IMGSZ` a `416` o `320`; puede perder precisión en objetivos pequeños. |
| Error de operadores de Torchvision | Reinstale `torch` y `torchvision` juntos desde el mismo índice CPU/CUDA. |
| No se selecciona un objetivo | Revise las clases reales del modelo, la selección en **Seguimiento** y los umbrales de confianza. |

Para trabajar temporalmente sin GPU, use `DETECTCAM_DEVICE=cpu` y reinicie.
Para comprobar el intérprete activo:

```bash
python -c "import sys; print(sys.executable)"
```

Debe apuntar a `.venv/bin/python`. En VS Code seleccione ese mismo intérprete.

## Comprobaciones de esta revisión

Se comprobaron sin conexión a cámara:

- carga de `.env` sin sobrescribir variables del entorno;
- resolución del modelo respecto de la raíz del proyecto;
- errores por `.env`, campos, booleanos y archivos faltantes;
- sintaxis/importación y versiones del entorno declarado;
- búsqueda de credenciales en archivos candidatos a publicación.

Estas comprobaciones no demuestran funcionamiento de RTSP, inferencia útil ni
movimiento físico.

## Pendiente con equipo físico

- conexión y reconexión RTSP con el modelo de cámara real;
- compatibilidad de los pesos y clases con imágenes reales;
- conexiones ISAPI y ONVIF;
- armado, límites, parada, dead-man y watchdog en un área controlada;
- dirección real de pan/tilt y comportamiento durante fallos de red;
- rendimiento medido en el equipo de despliegue.

No se afirma validación en campo ni una tasa de FPS. Consulte
[docs/ARQUITECTURA.md](docs/ARQUITECTURA.md) para el diseño comprobado en código.

## Pruebas de interfaz y control sin cámara

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Las pruebas usan imágenes sintéticas y un backend falso: comprueban el armado,
la exclusión entre modos, los límites, la caducidad de video/entrada, las
paradas, el registro y los botones. Las pruebas de widgets abren ventanas
breves; se omiten si no hay escritorio gráfico. No conectan una cámara real.

## Reducir el retraso del video

La vista en vivo lee el último cuadro RTSP sin esperar a YOLO. Por defecto se
usa el perfil de recepción de baja latencia con un hilo de decodificación.
Las cajas de detecciones se ocultan si superan 250 ms de edad local.

Consulte [la revisión y guía de latencia](docs/LATENCIA.md) para las mediciones
realizadas, ajustes de la web de la cámara y cómo medir el retraso real.
La indicación «Edad local» no mide el retardo completo cámara-pantalla.

## Preparar el primer commit

Antes de crear el commit, compruebe el entorno y las pruebas sin cámara:

```bash
source .venv/bin/activate
python -m pip check
python -m unittest discover -s tests -v
```

Desde la raíz del proyecto, si la carpeta aún no es un repositorio:

```bash
git init -b main
```

Compruebe que los archivos locales quedan excluidos y añada los archivos del
proyecto:

```bash
git check-ignore .env .venv/ model/best.pt x/best.pt test_onvif.zip test_onvif.7z
git add README.md requirements.txt .env.example .gitignore main.py test_onvif.py ptz/ tests/ docs/
git status --short
git diff --cached --stat
git diff --cached --check
git diff --cached
```

Revise que el contenido preparado incluya código, documentación, pruebas y la
plantilla `.env.example`. Los pesos (`*.pt`, `*.pth`, `*.onnx`, `*.engine`,
`*.safetensors`), `.env`, entornos virtuales, cachés, comprimidos, videos y
resultados locales están excluidos por `.gitignore`, incluso fuera de `model/`.
No incluya credenciales reales en la plantilla ni en ejemplos. `.gitignore`
no retira archivos que ya estén versionados; revise también la lista preparada.

Cuando la revisión esté completa:

```bash
git commit -m "Initial commit: Detectcam"
```

Si Git pide su identidad, configúrela para este repositorio con `git config
user.name "Tu nombre"` y `git config user.email "tu-correo@example.com"` y
repita el commit.

Para subirlo, cree un repositorio remoto vacío, sustituya la URL de ejemplo
por la suya y ejecute:

```bash
git remote add origin https://github.com/TU_USUARIO/Detectcam.git
git push -u origin main
```

El proyecto aún no incluye un archivo `LICENSE`; queda pendiente que su
responsable defina la licencia antes de presentarlo como software de libre uso.
