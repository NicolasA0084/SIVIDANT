# Instalación desde un clon nuevo

La rama contiene el código en su raíz. Los enlaces locales al proyecto anterior,
el entorno virtual, los pesos, los resultados y las credenciales no se publican.
Esta guía permite preparar recursos propios; la ejecución actual está diseñada
para Linux de escritorio y Python 3.12.

## Código y entorno

```bash
git clone --branch codex/ptz-propuesta https://github.com/NicolasA0084/SIVIDANT.git
cd SIVIDANT
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Instala los paquetes de sistema de tu distribución que proporcionan Tkinter
(`python3-tk` o equivalente para Python 3.12) y OpenGL (`libgl1` en Debian/Ubuntu).

Instala primero PyTorch y Torchvision compatibles entre sí y con tu dispositivo.
Para CPU, el índice oficial es `https://download.pytorch.org/whl/cpu`; para GPU
elige la variante según driver y hardware en el [selector oficial de PyTorch](https://pytorch.org/get-started/locally/).
Las versiones del entorno donde se hicieron las pruebas están registradas en
[entorno-verificado.json](entorno-verificado.json). Esta guía no instala ni
actualiza drivers automáticamente.

Después instala las dependencias directas:

```bash
python -m pip install -r requirements.txt
python -m pip check
./launch.sh --demo
./scripts/validate.sh
```

La demo y las pruebas no requieren modelo ni cámara reales. Las pruebas gráficas
necesitan una sesión de escritorio accesible; si no la hay se omiten explícitamente.

## Cámara y modelo

```bash
cp .env.example .env
mkdir -p model
```

Completa `DETECTCAM_CAMERA_IP`, `DETECTCAM_CAMERA_USER` y
`DETECTCAM_CAMERA_PASSWORD` en `.env`. Coloca los pesos existentes en
`model/best.pt` o indica su ubicación mediante `DETECTCAM_MODEL_PATH`.
El modelo y las credenciales se suministran por separado.

La plantilla inicia con `DETECTCAM_PTZ_HARDWARE_ENABLED=false`. Usa primero:

```bash
./launch.sh --check
./launch.sh --simulate
```

`--simulate` usa video y detección reales, sin crear el transporte físico PTZ.
Cuando configures `DETECTCAM_PTZ_HARDWARE_ENABLED=true`, el inicio normal
permanece desarmado; el movimiento requiere seleccionar modo y armar desde la UI.

## Recursos compartidos opcionales

En la instalación de desarrollo se reutilizan `.venv` y `model` mediante enlaces
a una carpeta hermana `ptz_test`; también se admite su `.env` como respaldo.
Esos recursos no son obligatorios si se proporciona un entorno propio, pesos
accesibles y configuración local. Un `.env` local con campos vacíos prevalece
sobre el respaldo: elimina las líneas que quieras heredar.

La ejecución soportada es desde el checkout mediante `main.py`/`launch.sh`.
`pyproject.toml` describe los módulos y dependencias, pero aún no distribuye
`settings.toml` como recurso de un wheel ni proporciona un comando global.
