# Seguimiento y control: contrato del núcleo

Este documento explica el comportamiento matemático de `core/` y sus valores de
arranque. El procedimiento de medición y los resultados pendientes se describen
en [PLAN_VALIDACION.md](PLAN_VALIDACION.md); el inventario de botones y atajos está
en [FUNCIONALIDADES.md](FUNCIONALIDADES.md).

El núcleo usa Python y biblioteca estándar, sin importar interfaz, detector ni
backend. `Tracker` pertenece al hilo de visión. `Controller` pertenece al hilo de
control y recibe un `TrackSnapshot` inmutable. Ninguna de estas clases ordena
movimientos físicos, espera intervalos ni consulta la cámara.

## Tiempos, unidades y evidencia

- Los tiempos son segundos del mismo reloj monotónico.
- `captured_at` identifica la recepción local del cuadro decodificado que se
  entregó al detector. No es el timestamp del sensor ni mide el tramo anterior
  del codificador, RTSP o red.
- `last_seen_at` identifica el cuadro de la última detección real del objetivo.
  Un cuadro sin detección actualiza `captured_at`, pero conserva `last_seen_at`.
- `frame_id` identifica el último cuadro observado; `last_seen_frame_id`, el
  último cuadro donde se detectó el objetivo.
- Centros, cajas y desplazamientos de fondo están en píxeles; las velocidades
  del tracker están en píxeles por segundo. X positivo apunta a la derecha y Y
  positivo hacia abajo. Los adaptadores traducen los signos de los motores.
- Pan/tilt de `MotionIntent` están normalizados al protocolo. No representan
  grados por segundo ni un desplazamiento físico calibrado.

Los IDs y timestamps de adquisición deben avanzar juntos. Repetir un cuadro,
entregar otro anterior o llamar `predict(now)` no crea evidencia nueva ni
prolonga la recuperación. Cambiar resolución o filtros reinicia la asociación.
Una selección vacía de clases significa que no hay objetivo permitido.

## Selección, identidad y estados

`Tracker.update(detections, frame_id, captured_at, frame_shape, min_confidence,
allowed_classes, camera_delta)` devuelve el estado completo. `camera_delta` es
el desplazamiento del fondo entre el cuadro anterior procesado y el actual:
`None` significa que no existe una estimación fiable; `(0, 0)` significa que sí
se estimó y no hubo desplazamiento. El estimador de flujo excluye las cajas para
reducir la influencia del objeto; una escena uniforme puede dejarlo sin evidencia.

| Estado | Significado |
| --- | --- |
| `IDLE` | No hay una detección elegible ni una identidad activa. |
| `ACQUIRING` | La opción `confirm` espera observaciones consecutivas. |
| `TRACKING` | Hay una detección asociada y el objetivo está confirmado. |
| `RECOVERING` | Falta el objetivo; todavía queda un plazo y una búsqueda acotados. |
| `EXPIRED` | La recuperación terminó; conservar memoria no autoriza movimiento. |

La opción predeterminada `acquisition_mode="fast"` confirma la primera detección
válida. `confirmation_hits=2` sólo se aplica al elegir `acquisition_mode="confirm"`:
dos observaciones consecutivas confirman; una observación sin asociación rompe
esa secuencia. No se impone una espera fija de tres cuadros.

Al seleccionar una identidad nueva se prioriza confianza, después proximidad al
centro y área. Para conservar una identidad existente se exige la misma clase y
se compara con la caja desplazada hacia la posición predicha. Si ninguna
detección se asocia, pero existe otro candidato válido, éste reemplaza al objetivo
inmediatamente; no espera a agotar su plazo de pérdida. La confianza y las clases
permitidas se aplican también a la asociación.

Esto es asociación geométrica de un objetivo, sin embeddings de apariencia ni
garantía semántica de identidad. Dos objetos de la misma clase que se cruzan
pueden confundirse; ese caso requiere medición y, si procede, otro asociador.

## Asociación y velocidades

Para una imagen de ancho `W`, alto `H` y diagonal `D`:

```text
distancia = ||centro_candidato - centro_predicho|| / D
score = 0.55 × IoU + 0.35 × (1 - min(1, distancia)) + 0.10 × confianza
```

Se admite la asociación si `distancia <= association_distance` **o**
`IoU >= min_iou`; entre los candidatos admitidos se elige el score mayor.
El desplazamiento predictivo para asociar se limita a `prediction_horizon_s`.

La velocidad aparente es el desplazamiento entre dos detecciones dividido por
su intervalo real de adquisición, no por un FPS supuesto. Se limita su magnitud
y se suaviza con un promedio exponencial dependiente de `dt`:

```text
alpha = 1 - exp(-dt / velocity_tau_s)
velocidad = anterior + alpha × (medida - anterior)
```

La primera medida de velocidad se aplica directamente. Cuando toda la cadena de
desplazamientos de fondo desde la detección previa es fiable, se separa una
velocidad del objeto compensando ese movimiento. La velocidad aparente incluye
la respuesta de nuestra propia cámara: no se usa como feedforward directo.

| Ajuste de tracking | Valor inicial | Unidad/efecto |
| --- | --- | --- |
| `association_distance` | `0.18` | Fracción de la diagonal. |
| `min_iou` | `0.03` | Solapamiento mínimo alternativo al gate de distancia. |
| `prediction_horizon_s` | `0.18` | Horizonte máximo para asociación. |
| `velocity_tau_s` | `0.12` | Suaviza velocidad, no el error de centrado. |
| `max_velocity_diagonals_s` | `2.0` | Máximo de velocidad medido en diagonales/s. |
| `recovery_s` | `0.85` | Plazo desde la última detección real. |
| `forget_s` | `1.20` | Memoria máxima antes de iniciar otra identidad. |

## Respuesta inmediata y parada

El controlador normaliza el error con respecto al centro de la imagen:

```text
ex = (centro_x - W/2) / (W/2)
ey = (centro_y - H/2) / (H/2)
orden = signo(error) × min(límite, max(mínimo_útil, ganancia × |error|))
```

Fuera de la zona muerta, usa `kp_acquire` si el error máximo de ambos ejes alcanza
`far_error`; cerca del centro usa `kp_track`. La parada por centrado se aplica
inmediatamente; no espera un cooldown ni una rampa descendente.

La zona muerta tiene histéresis por eje. En movimiento, entrar en `0.025` detiene
ese eje. Una vez detenido, necesita superar `0.025 + 0.015 = 0.040` para volver a
actuar. Esto evita alternar movimiento/parada ante pequeños cambios de la caja.

Ejemplos con los valores iniciales: un error de `0.5` solicita `0.65` tras aplicar
el límite; un error de `0.1` solicita `0.12`. Un error menor que la zona muerta
solicita cero. Son unidades del protocolo, no una predicción de velocidad real.

Sólo con velocidad compensada fiable se permite una anticipación corta:

```text
anticipación_x = clip(2 × velocidad_objeto_x / W × prediction_s, ±max_prediction)
```

Se requiere más de una observación. La anticipación se limita, y no se utiliza
durante la búsqueda del objetivo perdido.

| Ajuste de control | Valor inicial |
| --- | --- |
| `max_pan`, `max_tilt` | `0.65`, `0.65` |
| `kp_acquire`, `kp_track` | `1.8`, `1.2` |
| `far_error` | `0.25` |
| `dead_zone`, `dead_zone_hysteresis` | `0.025`, `0.015` |
| `min_command` | `0.07` |
| `prediction_s`, `max_prediction` | `0.06 s`, `0.08` de error normalizado |
| `target_fresh_s` | `0.30 s` |
| `max_frame_age_s` | `0.50 s` |
| `intent_ttl_s` | `0.12 s` |

El cálculo independiente del hilo de control vuelve a evaluar `now` aunque YOLO
no termine otra inferencia. Una snapshot que todavía diga `TRACKING` pasa a
tratamiento de recuperación cuando su detección supera `target_fresh_s`, y deja
de permitir movimiento al vencer su evidencia visual. `predict()` no disfraza
una inferencia bloqueada como una observación reciente.

## Recuperar el último punto, sin perseguir un vector indefinido

Durante la pérdida, el destino es la última posición observada, trasladada por
los desplazamientos fiables de fondo disponibles. La velocidad previa del objeto
no prolonga ese destino hacia el infinito.

Con compensación visual reciente y fiable se actúa hacia ese punto hasta
centrarlo o agotar el plazo. Si el punto cruza el centro, ese eje termina su
búsqueda: no invierte la dirección y sigue oscilando para buscar un objeto que no
se ve.

Sin esa evidencia visual se usa un presupuesto por eje:

```text
presupuesto_restante -= |orden_anterior| × duración_autorizada_anterior
duración_nueva <= presupuesto_restante / |orden_nueva|
```

`recovery_max_command=0.35` limita la intensidad; `recovery_budget=0.09` limita la
integral de orden por segundos. A intensidad máxima, ese presupuesto equivale a
unos `0.257 s` de orden. A menor intensidad dura más, respetando los demás plazos.
No representa una distancia física conocida. Una nueva detección real renueva
el presupuesto; los cuadros vacíos y los ticks del controlador no lo renuevan.

La recuperación puede durar **hasta** `recovery_s=0.85 s` desde la última
detección; puede terminar antes por centrado, presupuesto, evidencia vieja o una
parada del operador. El tracker y el controlador tienen su propio `recovery_s`;
se respeta el plazo menor. No basta con mantener vivo el video para buscar
indefinidamente el mismo objetivo.

Para cada intención en recuperación:

```text
expires_at = min(now + intent_ttl_s,
                 captured_at + max_frame_age_s,
                 last_seen_at + recovery_s,
                 deadline_del_tracker,
                 deadline_del_presupuesto_si_corresponde)
```

Un cuadro demasiado viejo, video no disponible, confirmación pendiente o
recuperación vencida produce una intención cero. Tampoco se acepta que un mismo
ID de cuadro cambie su timestamp para renovar artificialmente su validez.

## Entrega al runtime y pruebas sin cámara

`MotionIntent` comunica velocidad, motivo, fase, ID del cuadro y caducidad; el
runtime aplica además límites físicos y su propia caducidad máxima. Un latido de
la interfaz confirma presencia del operador y no amplía la evidencia del objetivo.
La orden cero pide STOP sin esperar el intervalo normal entre envíos.

El intervalo inicial de envío es `0.08 s`: un máximo teórico de `12.5` solicitudes/s
si red y cámara responden a tiempo. No es la tasa de inferencia ni una garantía
de respuesta mecánica. El TTL local limita autorizaciones; su efecto físico
depende de que STOP llegue a la cámara. ONVIF puede añadir un timeout del servidor
dentro del rango anunciado. Si su mínimo es mayor que la intención, ese timeout
también será mayor: el runtime publica el rango y sigue solicitando STOP al vencer
el TTL local. Soportar timeout de servidor no equivale a respetar exactamente el
plazo de la fuente. ISAPI continuo no ofrece ese TTL portable. La confirmación
HTTP/SOAP tampoco demuestra que el motor ya se detuvo.

La demo local genera cuadros y oclusiones con una dinámica ilustrativa; permite
inspeccionar transiciones y controles, no calibrar ganancias para la Hikvision.
El replay de [replay-ejemplo.json](replay-ejemplo.json) recorre observaciones con
tiempos y desplazamientos explícitos, sin detector ni comandos de red. El formato
y los argumentos de ejecución se documentan en el README y `--help`.

Las pruebas `test_tracking_core.py` y `test_control_core.py` verifican adquisición,
confirmación consecutiva, asociación, compensación, sustitución, timestamps,
caducidad, presupuestos y parada. No prueban precisión del modelo, identidad en
escenas reales ni tiempos físicos. Para esas conclusiones se necesita la sesión
comparativa del plan de validación.
