# Arquitectura de Detectcam

## Estado actual

Detectcam implementa adquisición RTSP, inferencia YOLO, selección de objetivo,
tracking temporal y seguimiento PTZ automático mediante movimiento continuo. El
código contiene adaptadores orientados a cámaras Hikvision. Esta revisión fue
estática y offline: no valida una cámara ni el movimiento físico.

La versión pública conserva un único ajuste: **SAFE**. No implementa todavía
búsqueda automática, PID con término integral, Smart Tracking, zoom automático
ni control automático mediante Position3D.

## Flujo de datos

~~~text
Cámara RTSP
    |
    v
RTSPStream -> YOLODetector -> TargetSelector -> TargetTracker
                                                   |
                                                   v
                                         PTZDecisionEngine
                                                   |
                                                   v
                                         PTZExecutionPolicy
                                                   |
                                                   v
                                         PTZCommandExecutor
                                                   |
                                                   v
                                  ISAPI PTZ -> ONVIF fallback
~~~

La información visual avanza de izquierda a derecha. El movimiento físico sólo
puede salir del PTZCommandExecutor; ninguna capa de detección o tracking conoce
las credenciales ni llama directamente a la cámara.

## Componentes

| Módulo | Responsabilidad |
| --- | --- |
| ptz/stream.py | Mantiene en un hilo el frame RTSP más reciente y reconecta ante fallos. |
| ptz/media.py | Guarda JPG en segundo plano y graba MP4 desde RTSP a los FPS de cámara, independientemente de Tk y YOLO. |
| ptz/detector.py | Carga best.pt una vez y convierte la salida de Ultralytics en detecciones internas. |
| ptz/target_selector.py | Filtra clases y confianza, y escoge un candidato cuando no existe un track vigente. |
| ptz/target_tracker.py | Conserva la identidad local mediante IoU, distancia y confianza; tolera pérdidas breves. |
| ptz/ptz_decision.py | Calcula error respecto al centro, filtra y genera HOLD, MOVE o candidato Position3D. |
| ptz/ptz_execution_policy.py | Evalúa antigüedad, confianza, frames visibles y funciones permitidas. |
| ptz/ptz_executor.py | Aplica armado, límites físicos, rate limit, dead-man, watchdog y parada. |
| ptz/ptz_backend.py | Define el contrato común y selecciona ISAPI u ONVIF. |
| ptz/isapi_ptz.py | Traduce velocidades normalizadas a ISAPI con HTTP Digest. |
| ptz/ptz_controller.py | Implementa movimiento continuo ONVIF como fallback. |
| ptz/ptz_tuning.py | Centraliza las constantes del perfil SAFE. |
| ptz/ptz_metrics.py | Registra telemetría opcional sin modificar decisiones ni comandos. |
| ptz/config.py | Carga `.env`, conserva la prioridad del entorno, valida y resuelve rutas. |
| main.py | Ejecuta el preflight y delega el pipeline a `ptz/main.py`. |
| ptz/main.py | Construye componentes y procesa visión en un hilo independiente. |
| ptz/ui.py | Interfaz Tkinter, video directo de RTSP, registro plegable y botones del operador. |
| ptz/preview.py | Anotaciones sobre el video; descarta cajas antiguas o de otra resolución. |
| ptz/operator_control.py | Reintenta la conexión inicial PTZ, serializa órdenes y arbitra modos manual/automático fuera del hilo gráfico. |
| ptz/tracking_settings.py | Comparte confianza y clases seleccionadas mediante instantáneas inmutables con revisión. |

## Convención de coordenadas

Entre las capas internas se usan coordenadas de pantalla normalizadas:

- pan negativo: izquierda;
- pan positivo: derecha;
- tilt negativo: arriba;
- tilt positivo: abajo.

Cada adaptador traduce esta convención al signo requerido por su protocolo. Esto
evita que la lógica de tracking dependa de ISAPI u ONVIF.

## Perfil SAFE

El perfil limita la respuesta antes de enviarla al backend:

| Parámetro | Valor |
| --- | ---: |
| Kp pan/tilt | 0.50 |
| Comando máximo pan/tilt | 0.35 |
| Velocidad física máxima pan/tilt automático | 0.30 |
| Velocidad máxima pan/tilt manual | 0.75 |
| Confianza mínima de movimiento inicial | 0.50 |
| Frames visibles mínimos | 3 |
| Intervalo mínimo entre comandos | 200 ms |
| Continuación tras pérdida temporal | 300 ms |
| Dead zone X/Y | ±0.10 |

El zoom automático permanece en 0.00. El zoom manual permite velocidades de
−0.30 a +0.30 cuando el backend anuncia continuous_zoom, con los mismos
interlocks de video, entrada renovada, parada y watchdog que pan/tilt.

La pestaña Seguimiento permite modificar la confianza durante la sesión. Se
aplica el mismo umbral a selección, decisiones y política de ejecución. El
umbral de inferencia baja si es necesario para que no oculte candidatos con
la confianza elegida. Las clases seleccionadas filtran tanto la adquisición
como la asociación del objetivo; ninguna clase seleccionada pausa el automático.

## Interlocks de seguridad

El control tiene dos compuertas independientes:

1. DETECTCAM_PTZ_HARDWARE_ENABLED=false evita incluso crear el backend.
2. Cuando se habilita hardware, el ejecutor sigue DISARMED hasta que el usuario
   pulsa Armar PTZ (o F2), sin un popup adicional.

La interfaz se crea antes de intentar conectar el backend PTZ. Si la cámara no
responde, OperatorControl reintenta la conexión en su hilo con una pausa de
2 segundos entre fallos; RTSPStream busca video de forma independiente. La UI
muestra la falta de señal y los eventos registran los intentos y la recuperación.
El ejecutor conectado inicia desarmado y descarta armados pendientes, por lo que
recibir señal no reanuda órdenes anteriores.

El modo inicial es manual. Cambiar de modo desarma y descarta las órdenes
manuales; el modo automático requiere volver a armar. La cruceta pasa por una
política manual que exige hardware, armado, backend compatible, video mostrado
reciente y renovación de la pulsación cada 250 ms. No exige ni simula un target.
La velocidad manual tiene un límite propio de 0.75; el seguimiento automático
conserva el límite físico del perfil. Ambos usan rate limit, dead-man y watchdog.
El teclado combina flechas y WASD; el zoom usa +/− del teclado numérico o la
barra de velocidad con retorno a cero. Al soltar, perder foco, minimizar o cambiar
de modo se descartan entradas pendientes. C/V disparan captura y grabación una
sola vez por pulsación, incluso con la repetición de teclado de X11.

MediaStore conserva una cola limitada de capturas y un único grabador. El
grabador utiliza el cuadro RTSP y su fecha de recepción local; escribe a los FPS
anunciados por la cámara (25 de respaldo), repitiendo cuadros cuando corresponde
para conservar la duración. Finaliza al perder señal, cambiar resolución, recibir
una parada o cerrar la aplicación. Nunca escribe video desde el callback de Tk
ni desde inferencia. Las capturas opcionalmente incluyen anotaciones; el MP4
siempre contiene video limpio y no incluye audio.

Además:

- d detiene y desarma;
- ESPACIO ejecuta la parada de emergencia;
- el dead-man detiene movimientos continuos demasiado largos;
- el watchdog detiene el sistema si deja de recibir actualizaciones válidas;
- targets o frames antiguos, baja confianza y pocos frames visibles producen
  HOLD o STOP;
- el cierre normal o una excepción intentan detener la cámara.

## Selección del backend

La fábrica acepta auto, isapi y onvif. En auto, ISAPI es el backend
principal. ONVIF sólo se usa si falla la conexión inicial de ISAPI. No existe
fallback después de enviar una orden: repetir un movimiento tras un timeout
podría duplicarlo.

## Configuración y secretos

`main.py` carga automáticamente `.env` desde la raíz antes de importar el pipeline.
Las variables ya presentes en el entorno tienen prioridad. `ptz/config.py` valida
los campos comunes, resuelve rutas relativas contra la raíz y configura redacción
de secretos en logging. `.env.example` documenta valores sin credenciales y
`.env` está excluido por `.gitignore`.

## Trabajo futuro

- evaluar aceleración con hardware AMD compatible;
- revalidar perfiles más rápidos con un protocolo comparable;
- diseñar y habilitar Position3D sólo con interlocks propios;
- evaluar autozoom;
- implementar búsqueda al perder definitivamente el objetivo;
- ampliar pruebas de integración sin cámara y con backends simulados;

## Interfaz y concurrencia

Tk sólo se usa desde el hilo principal. La vista previa lee el último cuadro
RTSP con su timestamp atómico, sin esperar a YOLO. El hilo de visión publica
sus anotaciones y datos en un buzón de una posición; el hilo de control es el único
que invoca al ejecutor para acciones manuales y automáticas. Las edades de las
decisiones crecen aunque YOLO no publique resultados; la edad del cuadro
procesado incluye el tiempo de inferencia. El control manual también comprueba
la edad del cuadro que la interfaz llegó a mostrar.

Los callbacks gráficos no hacen peticiones de red. El cierre detiene primero
el control físico, luego el stream y el procesamiento, y finalmente las
métricas. El registro gráfico usa la misma redacción de credenciales y limita
tanto su búfer como su texto visible a 2 000 líneas.

La recepción usa un hilo de decodificación FFmpeg y un sondeo acotado. Las
anotaciones caducan a los 250 ms; la vista en vivo puede continuar mientras
YOLO procesa. El timestamp de recepción es local y no mide la demora previa
del codificador de cámara ni de la red. Véase [LATENCIA.md](LATENCIA.md).
