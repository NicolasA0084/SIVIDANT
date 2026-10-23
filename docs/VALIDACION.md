# Resultados de validación · 21 de octubre de 2026

Revisión y ejecución en el entorno compartido Python 3.12.3. No se conectó RTSP,
ISAPI ni ONVIF a una cámara real y no se enviaron movimientos físicos. Los
transportes se validaron con sesiones/servicios falsos o la cámara sintética.

## Pruebas automatizadas

Comando: `./scripts/validate.sh`.

**103 pruebas descubiertas: 87 aprobadas, 16 omitidas por falta de pantalla Tk
accesible, cero fallos.** La compilación de sintaxis verificó 29 módulos de src.

| Grupo | Evidencia |
| --- | --- |
| Núcleo | Primera detección, corrección fuerte limitada, centrado, histéresis, pérdida temporal, nueva detección, vencimiento, tiempos/IDs inválidos y presupuesto de búsqueda. |
| Movimiento | MOVE bloqueado no bloquea snapshot ni solicitud STOP; STOP independiente y posterior; TTL, heartbeat, últimos comandos, fallos, cierre y recuperación. |
| Protocolos | XML/rangos ISAPI, ZRange, signos, Digest, sesión de parada separada; ONVIF simulado, rangos, puertos, zoom y mínimo de timeout remoto. |
| Integración | Primera detección llega a preview y control; inferencia lenta no rejuvenece evidencia; revisión vieja rechazada; cancelaciones concurrentes ganan al armado; UI viva conserva armado sin prolongar TTL. |
| Video/media/teclado | Último cuadro, reconexión, timestamps atómicos, anotaciones caducadas, JPG, MP4 reproducible, FPS/duración y pérdida de señal; WASD/flechas/zoom/atajos sin repetición. |
| Configuración/telemetría | Precedencia de entorno, modelo compartido, errores antes de conexión, replay, CSV/JSONL correlacionados y latencias. |

La omisión gráfica afecta pruebas que construyen widgets reales y comprueban
layout/foco/eventos Tk. Los manejadores de teclado, vista directa mientras el
detector está bloqueado y media sí se prueban sin pantalla. No se presenta la
comparación AST de métodos como sustituto de verificación visual.

## Comprobaciones completas

- `./launch.sh --check` y `--check --simulate`: configuración válida, modelo
  resuelto al archivo original y simulación física deshabilitada correctamente.
- `./launch.sh --replay docs/replay-ejemplo.json`: pan positivo de adquisición,
  recuperación con cuadros vacíos, nueva identidad con pan negativo y HOLD al
  vencer. No carga pesos ni necesita credenciales.
- `./launch.sh --demo --headless-seconds 10 --metrics`: ciclo completo con
  visión, control, runtime y telemetría sintéticos. Se registraron 232
  observaciones: 200 TRACKING, 25 RECOVERING y 7 EXPIRED; 95 despachos de
  movimiento. Es dinámica ilustrativa, no calibración de cámara real.
- Una segunda demo de 3 s verificó p50/p95 del backend simulado tras integrar el
  campo de latencia en las métricas. Sin eventos descartados en esas pruebas.
- Se comprobó que `.venv` y `model/best.pt` resuelven a los mismos recursos de
  `ptz_test`. `git status` de ese proyecto conserva sólo su `ptzcpy/` ya existente
  como no rastreado; no hay cambios en sus archivos rastreados.

## Modelo y coste del núcleo

Los pesos existentes se cargaron y ejecutaron sobre una imagen negra con acceso
Python de red deshabilitado. Primera inferencia ~1904 ms; segunda ~101 ms. No
estaba disponible CUDA para el proceso. Véase [ENTORNO.md](ENTORNO.md) y el
manifiesto de versiones/hash; no se infiere precisión con drones de esta prueba.

Tracker + controlador, sobre 5 000 muestras sintéticas de un objetivo tras
descartar 1 000 de preparación: p50 **0.028 ms**, p95 **0.034 ms**, máximo
**0.150 ms** en esa ejecución. No incluye inferencia, flujo óptico, red, Tk ni
mecánica; no es un benchmark de latencia total ni comparación Python/C++.

## Pendiente en equipo real

Validar aspecto y eventos gráficos en el escritorio; signos y velocidad según
zoom/FOV; latencia sensor-pantalla; respuesta y frenado mecánicos; seguimiento de
drones pequeños con pérdidas y cruces; capacidad de red y timeout ONVIF del
firmware. Los nuevos límites son parámetros iniciales, no una garantía de
centrado o recuperación física. El procedimiento está en
[PLAN_VALIDACION.md](PLAN_VALIDACION.md).
