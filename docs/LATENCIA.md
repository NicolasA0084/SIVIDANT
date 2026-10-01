# Latencia de video: revisión del 28 de septiembre de 2026

## Resultado comprobado

Se consultó la cámara mediante GET ISAPI y se compararon receptores RTSP.
No se cambiaron ajustes de la cámara, no se enviaron órdenes PTZ y no se
guardaron imágenes ni video.

Cámara: **Hikvision DS-2DT5432MWG-T**, firmware **V5.7.63**.
El canal 101 estaba en H.264, 1920×1080, 22 FPS, VBR máximo 2048 Kbps,
perfil Main, GOP/I-frame 22, smoothing 36, SmartCodec desactivado y sin audio.
En las muestras decodificadas hubo cuadros I/P y ningún B.

La versión anterior abría FFmpeg con **16 hilos**. El intento de usar
`CAP_PROP_BUFFERSIZE=1` devolvió false, por lo que no limitaba el búfer.

Para medir la mejora relativa se abrieron dos conexiones al mismo canal:
la configuración anterior y la nueva. Se emparejaron cuadros por hash de
píxeles submuestreados, se descartaron los primeros 20 pares y se compararon
**117 cuadros idénticos**. La nueva recepción se adelantó:

| Medida | Mejora relativa |
| --- | ---: |
| Mediana | 668.7 ms |
| Percentil 10 | 654.7 ms |
| Percentil 90 | 697.7 ms |

Es una prueba breve de recepción en este equipo; no incluye inferencia ni
pantalla y no demuestra una latencia absoluta menor de un segundo. Una lectura
posterior con el código integrado recibió 148 cuadros en una prueba de ocho
segundos, a **21.9 FPS** tras la conexión. El primer cuadro llegó a los **1273 ms**
desde el inicio; ese tiempo de arranque tampoco es la demora de cada cuadro.

## Cambios en Detectcam

- FFmpeg abre con un hilo de decodificación, `nobuffer`, `low_delay`,
  sondeo de 262144 bytes y análisis de 500000 microsegundos.
- Se mantiene TCP. La prueba más agresiva de sondeo estimó incorrectamente
  los FPS; se eligió el perfil moderado que reconoció 22 FPS.
- Tk presenta el último cuadro recibido directamente, sin esperar a YOLO.
- La inferencia continúa consumiendo sólo el último cuadro disponible.
- Cada cuadro se entrega junto a su ID y hora local bajo el mismo bloqueo.
- Las cajas del último resultado pueden acompañar al video nuevo durante
  un máximo de 250 ms desde la recepción del cuadro analizado; después se
  ocultan. No se reutilizan en otra resolución. La zona central permanece.
- El seguimiento automático conserva las edades del cuadro que analizó.
  Mostrar video nuevo no rejuvenece decisiones anteriores.

Los controles y los límites PTZ permanecen activos. Las pruebas de software
cubren captura/configuración, video con detector bloqueado, timestamps y
caducidad de anotaciones, además de las pruebas de control existentes.

La configuración ya está aplicada en `.env`:

```dotenv
DETECTCAM_RTSP_LOW_LATENCY=true
DETECTCAM_RTSP_TRANSPORT=tcp
DETECTCAM_RTSP_DECODER_THREADS=1
```

Reiniciar la aplicación es necesario. Si se ha exportado
`OPENCV_FFMPEG_CAPTURE_OPTIONS`, esa configuración externa tiene prioridad.
Para comparar con el perfil de recepción anterior, reiniciar con
`DETECTCAM_RTSP_LOW_LATENCY=false` y `DETECTCAM_RTSP_DECODER_THREADS=0`.
La vista en vivo seguirá siendo independiente de YOLO.

OpenCV permite fijar los hilos al abrir el decodificador; el efecto de
propiedades depende del backend. Véase la [referencia oficial de OpenCV](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html).
FFmpeg documenta que `nobuffer` reduce el almacenamiento durante el análisis
inicial y que el tamaño/tiempo de sondeo influye en la latencia. Véase
[FFmpeg: opciones de formato](https://ffmpeg.org/ffmpeg-formats.html#Format-Options).

## Qué poner en la web de la cámara

Primero probar el código nuevo conservando 1080p, para conocer el efecto del
cambio del receptor. Si aún se prioriza menos carga frente al detalle,
probar este perfil en **Configuración → Video/Audio → Video → Flujo principal
(normal)**. Son valores de prueba recomendados para esta cámara, no una
garantía del fabricante de llegar a un segundo.

| Campo | Valor recomendado |
| --- | --- |
| Tipo de video | Sólo video |
| Codificación | H.264 |
| H.264+ / Smart Codec | Desactivado |
| Resolución | 1280×720 para reducir carga; volver a 1920×1080 si se pierde detalle de objetivos pequeños |
| Velocidad de cuadros | 22 FPS, como ahora |
| Tipo de bitrate | CBR/Constante para la comparación con tráfico más predecible |
| Bitrate máximo | 2048 Kbps como punto de partida a 720p |
| Perfil | Main, como ahora; la muestra actual no usaba cuadros B |
| Intervalo I-frame / GOP | 22; probar 11 sólo si se busca recuperación más rápida tras pérdidas/reconexiones |
| SVC | Desactivado, como ahora |
| Smoothing / Suavizado | Mantener 36 inicialmente |

Las capacidades ISAPI de este equipo confirmaron 1280×720, 1280×960,
1920×1080 y 2560×1440, CBR/VBR y GOP de 1 a 400. La reducción a 720p puede
perjudicar la detección de drones lejanos; comparar con imágenes representativas.

Hikvision documenta esta ruta y los campos de video en su
[manual de Network Speed Dome](https://assets.hikvision.com/prd/public/all/doc/m000001163/UD13679B-B_Baseline_User-Manual-of-Network-Speed-Dome_V5.6.0_20221222.pdf).
El intervalo I-frame cuenta cuadros entre imágenes independientes; más I-frames
consumen más datos. Smoothing equilibra fluidez y calidad, y no está documentado
como un valor de latencia en segundos. Véase el
[manual oficial: I-frame y Smoothing](https://www.hikvision.com/content/dam/hikvision/products/S000000001/S000000002/S000000011/S000000013/OFR000059/M000043795/User_Manual/UD22560B-C_Network-Stable-Positioning-System_User-Manual_H5%2C-5.5.45_PDF1-TEST_en-US_20221221.PDF).

Si la web ofrece **Configuración → Local → Parámetros de visualización →
Rendimiento de reproducción**, seleccionar **Menor retardo / Shortest Delay**.
Ese ajuste afecta al reproductor web; no sustituye los ajustes RTSP de Detectcam.
Hikvision describe la prioridad de actualidad sobre fluidez en su
[manual de visualización local](https://assets.hikvision.com/prd/public/all/doc/m000070582/UD31398B_Baseline_Mobile-Network-Camera_User-Manual_V1.0.0_20221219.pdf).
El nombre y la disponibilidad del menú dependen del firmware.

Para la comparación, usar Ethernet en la misma LAN cuando sea posible y cerrar
otros visores de la cámara. Dejar TCP inicialmente. UDP es una prueba opcional
en una red estable mediante `DETECTCAM_RTSP_TRANSPORT=udp`; puede introducir
pérdidas y artefactos, por lo que no se activó ni se incluyó en las mediciones.

El substream 102 actual es 704×480, 30 FPS, VBR 1024 Kbps y GOP 50: reduce
detalle considerablemente. No se cambió el programa a ese canal automáticamente.

## Cómo verificar el objetivo de un segundo

1. Poner un cronómetro con centésimas frente a la cámara.
2. Fotografiar en una misma toma el cronómetro físico y su imagen en Detectcam.
3. Restar ambos tiempos. Repetir unas diez veces y comparar la mediana.
4. Repetir tras cinco minutos de uso para detectar acumulación de retraso.
5. Comparar primero el código nuevo a 1080p y después, si hace falta, a 720p.

**Edad local** es el tiempo desde que el decodificador entregó el cuadro al
programa. No incluye el sensor, codificador, red ni espera anterior en FFmpeg.
Un valor bajo no demuestra por sí solo que la cámara esté en tiempo real.
