# Entorno y modelo

En un clon nuevo se crea un entorno propio siguiendo [INSTALACION.md](INSTALACION.md).
El launcher admite tanto un `.venv` propio como uno enlazado.

En la instalación local original, el launcher utiliza el Python 3.12 existente en `../ptz_test/.venv`. `.venv` es
un enlace simbólico y `model` enlaza `../ptz_test/model`. No se creó otro venv,
no se instalaron paquetes y no se copiaron/modificaron pesos.

El inventario verificado de versiones, tamaño y SHA-256 del modelo está en
[entorno-verificado.json](entorno-verificado.json). `requirements.txt` y
`pyproject.toml` documentan dependencias directas; Torch/Torchvision de este
entorno son 2.14.0 / 0.29.0. La variante CPU/CUDA se debe conservar al migrar,
no sustituir por una instalación genérica sin comprobar el dispositivo.

La comprobación offline del 21 de octubre de 2026 cargó los pesos locales,
obtuvo las clases drone/airplane/helicopter/bird e hizo inferencia sobre una
imagen negra, con conexiones Python de red deshabilitadas. CUDA no estaba
disponible para este proceso. Primera inferencia aproximada: 1904 ms;
segunda: 101 ms. Es una comprobación funcional con imagen sintética, sin medir
exactitud ni un benchmark sostenido de video real.

La aplicación calienta el modelo antes de consumir cuadros de seguimiento.
Después mide inferencia más conversión de resultados. La transferencia de
resultados GPU se hace por lote de cajas, evitando .item() por cada atributo.

Si se mueve o elimina `ptz_test` en esa instalación compartida, hay que actualizar los dos enlaces o proporcionar
las rutas equivalentes. Los venv no suelen ser portables entre equipos o rutas
arbitrarias; los enlaces actuales están pensados para estas carpetas hermanas.

El clon nuevo usa su `.env` propio. La instalación compartida puede conservar las
credenciales en el `.env` anterior. No figuran en el manifiesto,
en documentación ni en la salida pública de `--check`.
