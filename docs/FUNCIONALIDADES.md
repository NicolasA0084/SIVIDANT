# Contrato de funciones y paridad de interfaz

La referencia funcional es [ptz/ui.py del commit anterior](https://github.com/NicolasA0084/SIVIDANT/blob/2019e19d114a1b464059d66e8458ca3a73c3edb8/ptz/ui.py) junto con sus pruebas, no la copia
antigua `ptzcpy`. La propuesta conserva el panel de operación y sus acciones, y
separa el seguimiento, el control, la red y la presentación. Esta lista describe
lo que debe comprobarse; una casilla vacía no significa que falte código, sino que
no se ha registrado aquí una verificación manual con cámara.

## Interfaz, video y eventos

| ID | Función que se conserva | Comprobación observable | Implementación y evidencia automatizada |
| --- | --- | --- | --- |
| V01 | Vista de cámara independiente de inferencia | Un detector lento no congela el video ni los controles. | `presentation/dashboard.py`, `adapters/stream.py`; pruebas de integración del pipeline. |
| V02 | Último cuadro disponible | No se reproduce una cola creciente de imágenes antiguas. | `adapters/stream.py`; `tests/parity/test_stream_latency.py`. |
| V03 | Video sin deformación | Se conserva la proporción al redimensionar la ventana y aparece un aviso sin señal. | `presentation/dashboard.py`; `tests/parity/test_ui.py`. |
| V04 | Reconexión de video | Cámara inicialmente ausente y pérdida posterior permiten recuperarse sin reiniciar el panel. | `adapters/stream.py`; `tests/parity/test_stream_latency.py`. |
| V05 | Anotaciones separadas del video original | Las cajas caducadas, futuras o de otra resolución no se dibujan; el cuadro original no se modifica. | `presentation/preview.py`; `tests/parity/test_stream_latency.py`. |
| V06 | Zona central opcional | La casilla permite mostrar u ocultar la referencia central. | `presentation/dashboard.py`, `presentation/preview.py`. |
| V07 | Registro plegable | Mostrar/ocultar Registro mantiene el video y sus controles. | `tests/parity/test_ui.py::test_video_log_toggle_and_live_data`. |
| V08 | Pestañas de datos | Se conservan Estado actual, Eventos, Archivos y Seguimiento. | `presentation/dashboard.py`; `tests/parity/test_ui.py`. |
| V09 | Eventos pausables y borrables | Pausar no bloquea visión/control; Limpiar borra el texto visible. El historial tiene límite. | `presentation/dashboard.py`; prueba del log acotado en `tests/parity/test_ui.py`. |
| V10 | Redacción de credenciales | Los mensajes visibles y el registro no muestran contraseñas ni credenciales de URL RTSP. | `config/environment.py`; `tests/parity/test_ui.py::test_bounded_log_redacts_credentials`. |
| V11 | Panel usable en ventana compacta | Emergencia permanece visible y los controles permiten desplazamiento. | `tests/parity/test_ui.py::test_emergency_remains_visible_in_compact_window`. |

## Control manual y modos

| ID | Función que se conserva | Comprobación observable | Evidencia |
| --- | --- | --- | --- |
| C01 | Arranque manual y desarmado | Abrir o reconectar no produce movimiento ni reanuda una pulsación previa. | Configuración, operador y pruebas con backend falso. |
| C02 | Armado explícito | Botón/F2 arma sólo si el hardware está disponible; no aparece un diálogo adicional. | `tests/parity/test_ui.py::test_arm_submits_directly_without_confirmation_dialog`. |
| C03 | Cambio manual/automático | Cambiar de modo detiene, descarta entradas y exige volver a armar. | Pruebas del operador. |
| C04 | Cruceta de ocho direcciones y parada central | Mantener mueve; soltar o salir del botón libera la orden. | `presentation/dashboard.py`; pruebas de foco/pulsación. |
| C05 | Teclado combinable | WASD y flechas permiten diagonales; liberar una tecla conserva las demás. Teclas opuestas se cancelan. | `tests/parity/test_keyboard.py::test_wasd_and_arrows_combine_and_releasing_one_keeps_the_other`. |
| C06 | Velocidad manual | Barra de 5 % a 75 %, inicio en 15 % y límite físico por eje. El automático conserva sus límites propios. | `tests/parity/test_ui.py::test_manual_speed_reaches_75_percent` y pruebas del operador. |
| C07 | Zoom manual | Barra/botones/teclado producen velocidad entre −0.30 y +0.30; al liberar vuelve a cero. Sin capacidad de zoom se deshabilita. | `tests/parity/test_keyboard.py`, `tests/parity/test_ui.py`. |
| C08 | Limpieza de entradas | Soltar, perder foco, minimizar, abrir selector de carpeta, cambiar de modo y cerrar cancelan entradas pendientes. | `tests/parity/test_keyboard.py`, `tests/parity/test_ui.py`. |
| C09 | Atajos compatibles con edición | Los atajos normales no interceptan escribir en campos ni combinaciones Ctrl/Alt; emergencia mantiene prioridad. | `tests/parity/test_keyboard.py`, pruebas de Espacio con foco en controles. |
| C10 | Estado del equipo | El panel diferencia simulación, conexión pendiente, desarmado, armado y fallo, y explica por qué se retiene un movimiento. | `presentation/dashboard.py`, `presentation/view_models.py`. |
| C11 | Reconexión explícita añadida | El botón Reconectar PTZ permite recuperar un fallo; cancela movimiento y vuelve desarmado. | `presentation/dashboard.py`, pruebas de recuperación del operador. |

### Atajos exactos

| Tecla | Acción |
| --- | --- |
| Flechas o W/A/S/D | Mantener pan/tilt; combinables. **D mueve a la derecha**. |
| + / −, incluidos Num + / Num − | Mantener zoom cuando está disponible. |
| F2 | Armar PTZ. |
| F3 | Desarmar PTZ. |
| Espacio o Esc | Emergencia: detener y desarmar. |
| C | Captura JPG, una vez por pulsación. |
| V | Iniciar/detener grabación, una vez por pulsación. |
| Q | Cerrar la aplicación con liberación de recursos. |

La repetición de teclado de X11 no debe generar capturas múltiples ni alternar
una grabación repetidamente. El texto antiguo que asignaba desarmado a `d` no
describe el código vigente: el desarmado es F3.

## Capturas y grabación

| ID | Función que se conserva | Comprobación observable | Evidencia |
| --- | --- | --- | --- |
| M01 | Captura a resolución de cámara | El JPG usa el cuadro completo, no la imagen reducida del lienzo. Rechaza señal ausente/antigua. | `tests/parity/test_ui.py::test_capture_uses_full_camera_frame_and_rejects_stale_images`. |
| M02 | Anotaciones opcionales en JPG | La opción de Archivos incorpora las anotaciones vigentes y respeta Mostrar zona central. | `presentation/dashboard.py`, `presentation/preview.py`. |
| M03 | Guardado en segundo plano | Escribir JPG no bloquea el panel; cola acotada y nombres con fecha/hora y sufijo único. | `adapters/media.py`; `tests/parity/test_media.py`. |
| M04 | MP4 limpio y sin audio | Se graba directamente del stream, sin cajas ni texto y sin esperar a YOLO. | `adapters/media.py`; prueba de MP4 reproducible. |
| M05 | Duración de grabación | Usa FPS anunciados por la cámara; 25 de respaldo. Repite cuadros cuando corresponde para preservar duración. | `tests/parity/test_media.py::test_recording_produces_playable_mp4_at_camera_fps_and_real_duration`. |
| M06 | Ciclo de grabación completo | Indicador REC/tiempo, Guardando al finalizar y mensaje final. Pérdida de señal, resolución nueva o cierre finalizan el archivo. | `tests/parity/test_media.py`. |
| M07 | Carpeta elegible | El selector muestra la ruta; se usan subcarpetas capturas/grabaciones y no se cambia de carpeta mientras se graba. | `tests/parity/test_media.py::test_cannot_start_without_signal_or_change_directory_during_recording`. |
| M08 | Media independiente del armado | Capturar/grabar requiere señal reciente, pero no armar el PTZ. Errores de disco/codificador llegan a Eventos. | `tests/parity/test_keyboard.py`, `tests/parity/test_media.py`. |

## Seguimiento, protección y telemetría

| ID | Función que se conserva | Comprobación observable |
| --- | --- | --- |
| T01 | Clases reales del modelo | La lista contiene las clases cargadas; seleccionar una clase no añade capacidad a los pesos. |
| T02 | Multiselección directa | Cada clic alterna una clase sin Ctrl; Todas y Ninguna tienen significado explícito. Ninguna detiene el automático. |
| T03 | Confianza editable | Barra de 1 % a 100 %; los cambios se aplican con Aplicar y duran esta sesión. |
| T04 | Aplicación coherente | La revisión de ajustes descarta resultados/órdenes anteriores y usa las clases elegidas tanto al adquirir como al asociar. |
| T05 | Identidad y pérdidas breves | El seguimiento conserva identidad temporal y distingue evidencia detectada de predicción. Una predicción no crea una detección nueva. |
| S01 | Video e intención recientes | Entrada, video mostrado, detecciones y latido de UI envejecen con reloj monotónico aunque YOLO esté bloqueado. |
| S02 | Límites físicos y temporales | Velocidad, cadencia de órdenes, caducidad/dead-man, watchdog, armado y prioridad de STOP se prueban con backend falso. |
| S03 | Adaptadores de cámara | ISAPI y ONVIF siguen separados de visión; el fallback automático ocurre durante conexión inicial, no repitiendo un movimiento tras timeout. |
| S04 | Cierre y fallos | Cerrar o fallar intenta detener el equipo y libera recepción, inferencia, media y telemetría. |
| D01 | Información de video | Resolución, estado de señal, FPS de inferencia, detecciones y edad local aparecen separados de la imagen. |
| D02 | Información de seguimiento y control | Objetivo, estado temporal, confianza, error, intención, movimiento confirmado, latencia backend y motivo de retención son distinguibles. |
| D03 | Métricas opcionales | Las muestras de visión/control pueden correlacionarse por identificador; escribir en disco no bloquea el control y las colas tienen límite. |

Las pruebas de widgets y teclado se portan en `tests/parity`. Las comprobaciones
del nuevo seguimiento, predictor y runtime se encuentran en las pruebas del núcleo
y del operador. [PLAN_VALIDACION.md](PLAN_VALIDACION.md) define la aceptación.

El seguimiento y el controlador se sustituyen deliberadamente. Los parámetros
históricos del perfil SAFE no se interpretan como ajustes del nuevo algoritmo;
se documentan los parámetros nuevos en `settings.toml`. El formato de telemetría
también cambia: no debe consumirse como si fuera el CSV anterior, ni presentarse
una métrica histórica no calculada como si fuera una medición actual.

## Lista de comprobación manual

- [ ] Recorrer todas las pestañas y controles con una fuente de video.
- [ ] Probar cada atajo de la tabla y su liberación; comprobar foco y minimización.
- [ ] Verificar capturas con/sin anotaciones y grabación reproducible.
- [ ] Probar confianza, varias clases, Todas, Ninguna y Aplicar.
- [ ] Comprobar conexión, reconexión, armado y emergencia en el equipo real.
- [ ] Comparar respuesta, latencia y recuperación usando el protocolo de validación.

Estas comprobaciones físicas no se sustituyen por pasar las pruebas sin cámara.
