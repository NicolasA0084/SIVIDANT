# Plan de validación y límites de la entrega

La validación se divide en software sin cámara, paridad visual y comportamiento
físico. Pasar un nivel no demuestra el siguiente. Este documento define el
procedimiento y los criterios; los resultados ejecutados de esta entrega se
registran en el README y en la salida de las pruebas.

## 1. Software sin cámara

Ejecutar desde `ptz-propuesta` con el entorno compartido ya preparado. No se
instalan paquetes ni se cargan pesos reales para estas pruebas; los componentes
de red/modelo se sustituyen por dobles y las imágenes son sintéticas.

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -v
```

Las pruebas de media escriben únicamente sus archivos temporales de comprobación.
Las pruebas de widgets necesitan servidor gráfico y pueden omitirse si no existe.
Una salida con pruebas omitidas se debe registrar como tal. Si Xvfb ya está
disponible, permite comprobar los widgets sin una pantalla física:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 xvfb-run -a .venv/bin/python -B -m unittest discover -s tests -v
```

No usar el arranque normal de la aplicación como sustituto de estas pruebas: la
simulación de movimiento puede seguir recibiendo RTSP real. Los modos de revisión
de configuración o replay sintético, cuando se usan, deben indicarse expresamente.

El contrato de la CLI distingue los siguientes modos. El README y `--help`
especifican los argumentos disponibles y el formato exacto de entrada del replay.

| Opción | Finalidad | Qué valida y qué puede abrir |
| --- | --- | --- |
| `--check` | Validar configuración y rutas compartidas. | Sólo preflight: no abre cámara ni carga el modelo. |
| `--simulate` | Operar la interfaz e inferencia sin movimiento físico. | Puede abrir RTSP real y cargar los pesos; no es una prueba sin cámara. |
| `--demo` | Probar la interfaz con imágenes/objetivo sintéticos. | Requiere pantalla para la ventana; no necesita cámara ni pesos reales. |
| `--replay` | Reproducir observaciones para seguimiento y control. | No envía comandos físicos; verifica el núcleo con los datos declarados, no la calidad del modelo real. |

| Área | Casos mínimos | Aceptación |
| --- | --- | --- |
| Configuración | Entorno > archivo local > compartido; modelo reutilizado; tipos/rangos inválidos; resumen sin secretos. | Error accionable antes de abrir cámara; no modificación del entorno virtual ni de los pesos. |
| Recepción | ID/frame/timestamp coherentes, perfil FFmpeg, opciones externas, pérdida y recuperación simuladas. | Último cuadro, sin cola creciente y con edad monotónica. |
| Vista previa | Detector bloqueado, caja antigua/futura, resolución distinta, imagen original intacta. | Video/controles siguen disponibles y se ocultan marcas no vigentes. |
| Teclado y widgets | Atajos de FUNCIONALIDADES, repetición X11, diagonales, zoom, foco, minimizar, ventana compacta. | Una acción por pulsación para C/V; liberar entrada detiene intención manual; emergencia accesible. |
| Media | JPEG íntegro, nombres únicos, MP4 reproducible, FPS/duración, pérdida de señal y fallos de codificador/disco. | Se cierra el archivo y se informa el error sin bloquear el control. |
| Ajustes en sesión | Clases reales, Todas/Ninguna, confianza y revisión aplicada durante inferencia. | Se descarta evidencia previa y no se mueve por una clase excluida. |
| Seguimiento | Movimiento, cruce de distractor, pérdida breve/larga, asociación y reacquisición. | No cambia de identidad por un candidato incompatible; predicción caduca. |
| Movimiento de cámara | Traslación conocida, cuadro antiguo y cielo uniforme. | Se estima la traslación sintética dentro de tolerancia y se rechaza evidencia insuficiente. |
| Control | Antigüedad del cuadro incluye inferencia; pérdidas breves; límites y rampas; saturación. | Comandos finitos/acotados y retención explícita cuando falta evidencia. |
| Runtime PTZ | Armado, modos, backend lento/fallido, STOP prioritario, watchdog, dead-man y cierre. | No hay movimiento desarmado; órdenes antiguas se descartan y una reconexión no rearma. |
| Telemetría | Cola saturada, disco lento/fallido y correlación de IDs. | La muestra puede perderse con contador; el control no espera al disco. |

Las pruebas portadas bajo `tests/parity` conservan comprobaciones del panel,
capturas, teclado y stream anterior. No se exige que el nuevo algoritmo produzca
las mismas velocidades que el antiguo; se exige conservar las acciones del
operador y cumplir los límites nuevos declarados.

`tests/test_application.py` comprueba además la integración: primer objetivo en
la vista, edad conservada tras inferencia lenta, revisiones de ajustes en vuelo,
presencia de UI mientras el PTZ está armado sin moverse, cancelación concurrente
con armado y lectura directa del video mientras YOLO está bloqueado. Sus dobles
de red y detector son locales; esa última comprobación tampoco requiere Tk visible.

## 2. Paridad visual y fuentes grabadas

Usar el [contrato de funciones](FUNCIONALIDADES.md) como lista de aceptación. Una
fuente grabada debe conservar timestamps coherentes y reproducirse a ritmo real;
procesarla sin esperas mide rendimiento máximo, no latencia de operación.

1. Recorrer Estado actual, Eventos, Archivos y Seguimiento y todos sus botones.
2. Probar atajos, selección de clases y ajustes mientras la inferencia se retrasa.
3. Capturar con/sin anotaciones y reproducir el MP4 resultante.
4. Probar secuencias de movimiento uniforme, aceleración, cambios de sentido,
   oclusión, distractores, salida del encuadre y variaciones de escala.
5. Repetir con textura de fondo y cielo uniforme; distinguir compensación fiable
   de predicción sin compensación.

Registrar pérdidas de identidad, tiempo de recuperación y errores de centrado.
Un replay sintético verifica transiciones y límites; no demuestra precisión del
modelo, recuperación de un dron real ni respuesta mecánica.

## 3. Medición comparativa en la cámara real

Esta fase queda pendiente hasta ejecutar una sesión con el equipo. Conservar el
mismo modelo, escena, resolución, canal, dispositivo de inferencia y carga del
equipo al comparar `ptz_test` con `ptz-propuesta`. Cambiar sólo una variable por
comparación y registrar los ajustes efectivos de ambas versiones.

| Tramo/resultado | Medida y método |
| --- | --- |
| Sensor a pantalla | Cronómetro visible para la cámara y fotografía simultánea del cronómetro físico y del panel. Repetir al inicio y tras varios minutos. |
| Recepción local | Edad desde entrega del decodificador; no incluye sensor/codificador/red anterior a esa entrega. |
| Inferencia | Duración y FPS completados con el modelo/dispositivo reales; incluir calentamiento por separado. |
| Decisión y envío | Timestamps e IDs desde cuadro analizado hasta intención y comando confirmado. Registrar p50/p95 y muestras descartadas. |
| Backend | Duración de la petición; una respuesta HTTP/SOAP no confirma por sí sola el inicio o fin del movimiento físico. |
| Mecánica | Comparar vídeo del movimiento real con la orden; dirección, retardo, aceleración y asentamiento. |
| Seguimiento | Error de centrado, sobrepaso, oscilación, tiempo fuera de encuadre, identidad y reacquisición. |
| Sesión larga | RAM, CPU/GPU, colas, muestras omitidas, latencia y recuperación después de desconexión. |

Separar FPS de recepción, FPS de inferencia y frecuencia de comandos. Más FPS o
una edad local pequeña no bastan para afirmar menor retraso extremo a extremo.
Para medir el efecto del controlador, usar trayectorias repetibles y comparar
además el error de seguimiento. Una velocidad mayor puede empeorar oscilaciones.

Antes de evaluar seguimiento automático se verifican en el equipo real los
signos pan/tilt, la parada, el desarmado, el zoom manual y los límites declarados.
Los valores de arranque son puntos iniciales que requieren calibración de esa
cámara y lente; no representan una respuesta física ya medida.

## Registro de cada comparación

- Revisión de código y configuración efectiva, sin credenciales.
- Hash del modelo, clases, tamaño de inferencia y umbrales.
- Versiones de Python, OpenCV, Torch/Torchvision y Ultralytics; CPU/GPU y driver.
- Firmware y ajustes de cámara, transporte, red y número de visores abiertos.
- Duración, escenario, calentamiento, p50/p95, fallos y pruebas omitidas.
- Video o trazas que permitan reproducir el caso cuando se hayan recogido.

No se atribuye una mejora de latencia a C++, a la nueva estructura o al predictor
sin completar esta comparación. El software usa ya operaciones nativas de
OpenCV/PyTorch; el perfil medido determina el siguiente trabajo de optimización.
