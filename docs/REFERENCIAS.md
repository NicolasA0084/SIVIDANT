# Referencias y decisiones de diseño

Consulta de fuentes primarias: 21 de octubre de 2026. Los repositorios siguientes
se revisaron como referencias de diseño; no se instalaron, ejecutaron ni se copió
su código. La propuesta reutiliza el entorno, pesos y componentes de adquisición
e interfaz existentes. Las ideas incorporadas son implementaciones locales; no
equivalen a integrar los algoritmos completos de estos proyectos.

## Proyectos comparables

| Proyecto y licencia visible | Idea útil para Detectcam | Alcance y límite |
| --- | --- | --- |
| [ByteTrack, repositorio de los autores](https://github.com/ifzhang/ByteTrack). [MIT](https://github.com/ifzhang/ByteTrack/blob/main/LICENSE). | Separar adquisición de un track y asociación temporal; usar predicción y evidencia de menor confianza para recuperar continuidad. El [tracker original](https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/byte_tracker.py) contiene predicción Kalman y dos fases de asociación. | Es tracking de múltiples objetos, no un controlador PTZ ni una validación de drones lejanos. La propuesta usa seguimiento de un objetivo y criterios propios; no implementa ByteTrack completo. |
| [Norfair, repositorio de Tryolabs](https://github.com/tryolabs/norfair). [BSD-3-Clause](https://github.com/tryolabs/norfair/blob/master/LICENSE). | Separar movimiento aparente del fondo y movimiento del objetivo antes de predecir. La [demo de cámara móvil](https://github.com/tryolabs/norfair/tree/master/demos/camera_motion) compara traslación y homografía. | Un fondo estático con textura favorece la estimación. El cielo uniforme, nubes o cambios fuertes pueden hacerla poco fiable. Detectcam debe declarar ese resultado como no fiable y continuar con una política conservadora, sin inventar movimiento del fondo. |
| [Ptz_camera_control, repositorio del autor](https://github.com/thenameisAnurag/Ptz_camera_control). No aparece una licencia explícita en el árbol raíz consultado. | Comparar la separación captura/inferencia/PTZ y el descarte de comandos viejos con un proyecto que también usa RTSP, YOLO, teclado y ONVIF. | Su README incluye un diagnóstico con movimiento programado. No se ejecutó. Sus afirmaciones de FPS o ausencia de retraso no son mediciones de Detectcam. Se toma sólo como referencia conceptual, sin reutilizar código. |

Las licencias indicadas corresponden a los archivos consultados; no cubren
automáticamente pesos, videos ni todas las dependencias de esos repositorios.
Si se integra código externo en el futuro, se debe identificar la revisión exacta
y conservar los avisos que correspondan a esa integración.

## Python y cálculo nativo

Python coordina la aplicación, el estado del operador y los mensajes entre hilos.
OpenCV expone desde Python funciones implementadas en C++; sus enlaces generan
el puente hacia esas funciones. Véase la [explicación oficial de los bindings de
OpenCV](https://docs.opencv.org/5.0/py_tutorials/py_bindings/py_bindings_basics/py_bindings_basics.html).

PyTorch utiliza ATen para operaciones de tensores y dispone de implementaciones
CPU/GPU accesibles desde las interfaces Python y C++. La disponibilidad real de
CUDA depende del entorno y del dispositivo configurado. Véase la [documentación
oficial de la API C++/ATen de PyTorch](https://docs.pytorch.org/cppdocs/).

De ello se infiere que reescribir pequeñas operaciones de decisión a C++ no
elimina el tiempo del codificador de cámara, RTSP, decodificador, inferencia,
petición de control o mecánica. Primero se instrumentan esos tramos. Sólo una
medición que identifique una sección Python dominante justificaría portar esa
sección o mover un trabajo intensivo a otro proceso. No se afirma una aceleración
medida por cambiar la estructura o introducir predicción.

## Aplicación concreta de las ideas

- El núcleo trabaja con datos y timestamps, sin importar Tk, HTTP ni el modelo.
- La captura publica el último cuadro y la interfaz no espera a la inferencia.
- La continuidad del track y el permiso de movimiento se evalúan por separado.
- La compensación de fondo aporta una estimación junto con su fiabilidad.
- La predicción tiene horizonte limitado; no equivale a confirmar presencia.
- Un único emisor serializa órdenes y descarta evidencia caducada.
- La telemetría distingue intención visual y comando realmente enviado.

Estas decisiones deben evaluarse con secuencias representativas de drones:
objetivos de pocos píxeles, cruces, oclusión, pérdida de detección, pan/tilt,
variaciones de escala y cielo sin textura. Los ejemplos externos sirven para
formular hipótesis, no para trasladar cifras de exactitud o FPS.
