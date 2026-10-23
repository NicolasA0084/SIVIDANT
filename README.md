# Detectcam 2 · propuesta de seguimiento PTZ

Versión de seguimiento en la rama `codex/ptz-propuesta` de SIVIDANT. Conserva la interfaz anterior y cambia
el seguimiento y la ejecución de órdenes para corregir el comportamiento de
«detecta, da un tirón y se detiene». No modifica `ptz_test`.

## Ejecutar

Desde la raíz de esta rama. Para un clon nuevo, prepara primero el entorno,
la configuración y los pesos siguiendo [INSTALACION.md](docs/INSTALACION.md).
La instalación local original puede seguir reutilizando sus recursos compartidos.

```bash
./launch.sh --check       # preflight sin modelo en memoria ni conexiones
./launch.sh --demo        # interfaz completa con cámara/motores sintéticos
./launch.sh --simulate    # RTSP y YOLO reales; ningún transporte PTZ
./launch.sh               # cámara configurada; inicia manual y desarmado
```

`launch.sh` utiliza `.venv/bin/python`, tanto si es un entorno propio como si es
el enlace local al entorno existente. En la instalación de desarrollo original,
`.venv -> ../ptz_test/.venv` y `model -> ../ptz_test/model` reutilizan recursos sin
duplicarlos. También funciona `.venv/bin/python -B main.py`.

La configuración se lee del `.env` local; si existe `../ptz_test/.env`, se usa
como respaldo. Las variables del proceso tienen prioridad sobre ambos. La
plantilla [.env.example](.env.example) deja hardware deshabilitado. Al habilitarlo,
el arranque normal consulta capacidades y envía STOP de inicialización, pero
**no arma ni inicia seguimiento por sí solo**. `--simulate` evita crear el transporte físico.

Para seguir drones: inicia, selecciona **Seguimiento automático** y pulsa
**Armar PTZ [F2]**. Cambiar de modo desarma. **Espacio/Esc** paran y desarman;
**F3** desarma. Clases iniciales: `drone`, configurable en `settings.toml` y en la
pestaña Seguimiento. Otras clases del modelo siguen disponibles para seleccionar.

## Cambios de comportamiento

- Primera detección válida: corrección inmediata, sin esperar tres cuadros.
  El modo opcional `confirm` permite exigir observaciones consecutivas.
- Centrado rápido lejos del objetivo; menor ganancia al acercarse; zona central
  más pequeña y orden mínima útil fuera de ella.
- Una pérdida breve mantiene una búsqueda acotada hacia la última posición.
  No se renueva el plazo por recibir imágenes vacías.
- Si hay otra detección válida que no se asocia al objetivo perdido, pasa a ella
  de inmediato, respetando clases y confianza seleccionadas.
- El movimiento del fondo se estima con OpenCV para transportar la última
  posición a la imagen actual. En cielo sin textura puede no haber estimación:
  en ese caso sólo se permite un presupuesto corto de búsqueda.
- El reloj del control funciona independientemente de YOLO. Fechar la evidencia
  al recibir el cuadro evita rejuvenecer imágenes tras una inferencia lenta.
- Red fuera de bloqueos de estado, último comando pendiente, caducidad, STOP
  prioritario por una conexión separada y STOP posterior a un MOVE en vuelo.
- Telemetría asíncrona en JSONL/CSV con cuadro y comando correlacionados.
- Botón **Reconectar PTZ**, siempre con recuperación desarmada.

Se conservan video vivo, registro plegable, pestañas, filtros, teclado/cruceta,
velocidad y zoom manual, capturas anotadas opcionales, grabación MP4 limpia y
selección de carpeta. La matriz completa está en
[FUNCIONALIDADES.md](docs/FUNCIONALIDADES.md).

## Ajustes de la propuesta

Las ganancias nuevas están en [settings.toml](settings.toml). Los parámetros
antiguos `SAFE`, `MAX_MISSED_FRAMES`, ganancias y límites antiguos de `.env`
no gobiernan este núcleo. Conexión, dispositivo, resolución de inferencia y
confianza mínima inicial se siguen reutilizando desde el entorno existente.

| Ajuste | Valor inicial | Significado |
| --- | ---: | --- |
| `control.max_pan/max_tilt` | 0.65 | Límite automático normalizado. |
| `control.kp_acquire/kp_track` | 1.8 / 1.2 | Corrección lejana / cercana. |
| `control.dead_zone` | 0.025 | Tolerancia respecto a la semidimensión de imagen. |
| `control.min_command` | 0.07 | Orden mínima fuera de la zona e histéresis. |
| `runtime.command_interval_s` | 0.08 s | Máximo teórico 12.5 envíos/s; depende de red. |
| `tracking/control.recovery_s` | 0.85 s | Plazo máximo desde la última detección. |
| `control.recovery_budget` | 0.09 | Integral de orden por eje cuando falta odometría fiable. |
| `control.max_frame_age_s` | 0.50 s | Edad máxima de evidencia local. |
| `control.intent_ttl_s` | 0.12 s | Vigencia corta de cada intención, limitada por su fuente. |

Son valores de diseño iniciales: no se han calibrado con movimiento físico de
tu cámara. 0.65 no representa grados/s ni garantiza 65 % de la velocidad real.
Una recuperación puede terminar antes de 0.85 s por llegar al punto, agotar el
presupuesto, perder señal o caducar el cuadro. En la interfaz puedes inspeccionar
el motivo, la fase, la edad y la orden que se envía.

## Estructura

```text
main.py                       arranque local
settings.toml                 parámetros del nuevo seguimiento y transporte
src/detectcam/
  core/                       modelos, asociación, control; sin I/O ni dependencias
  application/                composición, pipeline, operador, replay, telemetría
  adapters/                   RTSP, YOLO, flujo óptico, media y cámara sintética
  runtime/                    ISAPI/ONVIF, colas, plazos, STOP y reconexión
  presentation/               Tkinter, anotaciones y datos de presentación
  config/                     configuración validada y redacción de secretos
tests/                        lógica, integración, transportes falsos y paridad UI
docs/                         arquitectura, seguimiento, validación y referencias
```

Python coordina estado e interfaz. Decodificación, imágenes, flujo óptico e
inferencia utilizan los motores nativos C++/CPU/CUDA existentes. No se añadió
una extensión C++ para unas pocas operaciones de control: en una prueba local
de 5 000 muestras sintéticas, tracker + controlador tuvieron p50 de 0.028 ms y
p95 de 0.034 ms. Esa prueba no mide video, inferencia, red ni movimiento físico.
Las decisiones de lenguaje están explicadas en
[ARQUITECTURA.md](docs/ARQUITECTURA.md) y [REFERENCIAS.md](docs/REFERENCIAS.md).

## Verificación y diagnóstico

```bash
./scripts/validate.sh
./launch.sh --replay docs/replay-ejemplo.json
./launch.sh --demo --headless-seconds 10 --metrics
./launch.sh --metrics
```

El replay comprueba pérdida, recuperación, cambio de objetivo y vencimiento
sin modelo ni cámara. El demo completo genera imágenes y responde a comandos
simulados; no demuestra precisión de detección ni calibración de una PTZ real.

Con `--metrics`, `runs/<fecha>/` contiene `metadata.json`, `events.jsonl`,
`events.csv` y `summary.json`. Los eventos distinguen aceptación, despacho,
respuesta y parada; una respuesta HTTP/SOAP **no mide la respuesta mecánica**.
Las estadísticas usan ventanas acotadas; si una cola se llena, se descartan
muestras antes que bloquear el control. El formato sustituye el informe antiguo;
no presenta recenter/overshoot mecánico como si estuvieran medidos.

Resultados y límites verificados: [VALIDACION.md](docs/VALIDACION.md).
Procedimiento físico pendiente: [PLAN_VALIDACION.md](docs/PLAN_VALIDACION.md).

## Límites que importan al seguimiento

- `captured_at` es recepción local tras decodificar, no tiempo del sensor.
- El último punto de imagen no equivale a una posición absoluta calibrada. Sin
  fondo fiable, sólo existe una búsqueda de presupuesto limitado.
- Dos drones de la misma clase pueden intercambiar identidad en un cruce.
- ISAPI continuous no anuncia un TTL portable. En ONVIF se respeta el rango
  anunciado; si exige mínimo 1 s, ese plazo remoto puede superar una intención
  local de 0.12 s. STOP local sigue siendo necesario.
- Una red caída o peticiones que se cruzan en la cámara impiden garantizar el
  instante de detención física. Se expone STOP confirmado/no confirmado y se
  repite STOP tras el retorno de un movimiento que estaba en vuelo.
- Este entorno no ofrece pantalla Tk accesible; las pruebas gráficas se omiten
  explícitamente. Los manejadores de teclado y la vista con YOLO bloqueado sí
  se verifican sin pantalla.

Detalles de fórmulas, estados y caducidad: [SEGUIMIENTO.md](docs/SEGUIMIENTO.md).
