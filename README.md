[README.md](https://github.com/user-attachments/files/31009035/README.md)
# 🎯 SafeStepAI — Guía de Voz para Detección de Objetos

> Reconocimiento de objetos en tiempo real, narrado en español, con interfaz gráfica y compatibilidad con cámara web, imágenes y video.

---

## 📖 Descripción del proyecto

**SafeStepAI** es una aplicación de escritorio pensada como apoyo de orientación para personas con baja visión, o como sistema de asistencia mientras el usuario se desplaza. Combina visión por computadora con síntesis de voz: detecta lo que aparece frente a la cámara y lo describe en voz alta, en español.

La app reconoce hasta 80 categorías distintas (personas, vehículos, animales, mobiliario, entre otras), calcula en qué zona horizontal se encuentra cada objeto (izquierda, centro, derecha) y qué tan cerca está (muy cerca, cerca, media distancia, lejos), y arma frases habladas usando los artículos gramaticales correctos.

Incluye dos formas de uso:
- **`deteccion_objetos.py`**: módulo/script de consola, pensado para reutilizarse como backend de una API web.
- **`app_gui_2.py`**: interfaz gráfica de escritorio con controles en tiempo real, vista previa de la cámara y panel de detecciones.

---

## 🧰 Stack técnico

### 🤖 [Ultralytics / YOLOv8](https://github.com/ultralytics/ultralytics)
Es el motor de detección. YOLOv8 procesa cada frame en una sola pasada por la red neuronal, lo que permite identificar objetos con buena precisión sin sacrificar velocidad. El proyecto permite elegir entre tres variantes del modelo:
- `yolov8n` — la más liviana y rápida, con menor precisión.
- `yolov8s` — punto intermedio entre velocidad y precisión.
- `yolov8m` — la más precisa, a costa de mayor tiempo de inferencia.

La primera vez que se ejecuta, el modelo se descarga solo.

---

### 📷 [OpenCV (`opencv-python-headless`)](https://opencv.org/)
Se encarga de toda la parte de captura y manejo de imagen:
- Lectura de frames desde webcam o archivo de video.
- Conversión entre espacios de color (BGR/RGB).
- Renderizado de las cajas delimitadoras sobre el video.
- Guardado de capturas puntuales en disco.
- Detección de las cámaras disponibles en el equipo (DirectShow/MSMF).

---

### 🖼️ [Pillow (`PIL`)](https://pillow.readthedocs.io/)
Se usa para preparar cada frame antes de mostrarlo en la interfaz:
- Ajusta el tamaño de la imagen al espacio disponible en el canvas.
- Convierte los arrays de NumPy al formato que Tkinter puede renderizar (`ImageTk`).

---

### 🔊 [gTTS (Google Text-to-Speech)](https://gtts.readthedocs.io/)
Transforma en audio las frases de guía que genera el sistema, apoyándose en el servicio de Google. Se configuró con acento mexicano (`tld="com.mx"`) porque suena más natural para la narración. El resultado se guarda como MP3 temporal antes de reproducirse.

---

### 🎵 [pygame](https://www.pygame.org/)
Solo se usa para reproducir los MP3 generados por gTTS: carga el audio, lo reproduce y espera a que termine antes de continuar (`get_busy()`).

---

### 🔢 [NumPy](https://numpy.org/)
Sostiene el manejo numérico de los frames, que llegan como arrays con forma `(H, W, 3)`. También se usa para crear un frame vacío con el que se "calienta" el modelo al iniciar.

---

### 🖥️ [Tkinter](https://docs.python.org/3/library/tkinter.html)
Es la base de la interfaz gráfica en `app_gui_2.py`. Con ella se construyó:
- Un panel lateral con controles (sliders, comboboxes, checkboxes).
- Un canvas central con el video en vivo y las detecciones dibujadas.
- Un panel inferior que muestra el detalle de las detecciones y el texto narrado.
- Botones para iniciar, detener, capturar y salir.

---

### ⚡ [PyTorch (`torch`)](https://pytorch.org/) *(opcional)*
El sistema revisa si hay una GPU disponible (`cuda`) para acelerar la inferencia de YOLO. Si `torch` no está instalado, no pasa nada: el modelo corre en CPU sin necesidad de configuración extra.

---

### 📊 [Matplotlib](https://matplotlib.org/)
Se usa en la versión de consola (`deteccion_objetos.py`) para desplegar imágenes estáticas con las detecciones ya dibujadas.

---

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

> **Nota:** el soporte de GPU para PyTorch se instala aparte, según el sistema operativo y la versión de CUDA que tengas. Ver [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 📂 Estructura del proyecto

```
SafeStepAI/
├── deteccion_objetos.py   # Lógica de detección (consola + módulo reutilizable)
├── app_gui_2.py           # Interfaz gráfica de escritorio
├── requirements.txt       # Dependencias del proyecto
├── yolov8n.pt             # Pesos del modelo YOLO (se descargan solos)
├── resumen_video.mp3      # Último audio-resumen generado
└── README.md              # Este archivo
```

---

## 🖥️ Uso — Interfaz gráfica

```bash
python app_gui_2.py
```

Al abrir, la app muestra un panel lateral de configuración junto al visor de video en vivo.

### Controles disponibles:
| Control | Descripción |
|--------|-------------|
| **Modo** | Cámara web / Imagen fija / Video |
| **Cámara** | Selector de cámara con escaneo automático |
| **Espejo** | Invierte horizontalmente la imagen de la webcam |
| **Confianza mínima** | Umbral de detección (5%–95%) |
| **Modelo YOLO** | Nano, small o medium |
| **Dispositivo** | Auto / CPU / GPU (según disponibilidad) |
| **Filtrar objetos** | Restringe la detección a categorías puntuales (ej: `persona, auto`) |
| **Narración activa** | Enciende/apaga la salida de voz |
| **Intervalo de narración** | Segundos entre cada anuncio (1s–15s) |
| **Resolución de inferencia** | 320 / 416 / 640 px (velocidad vs. precisión) |
| **📷 Captura** | Guarda el frame actual como PNG |

---

## ⌨️ Uso — Script de consola

### 1. Imagen fija

```bash
python deteccion_objetos.py --source foto.jpg --type image
```

### 2. Video

```bash
python deteccion_objetos.py --source video.mp4 --type video
```

### 3. Cámara web (opción por defecto)

```bash
python deteccion_objetos.py --type webcam
```

### Parámetros

| Parámetro | Corto | Descripción | Default |
|-----------|-------|-------------|---------|
| `--source` | `-s` | Ruta de imagen/video o índice de cámara | `None` (webcam 0) |
| `--type` | `-t` | `image` \| `video` \| `webcam` | `webcam` |
| `--output` | `-o` | `narrar` \| `mostrar` \| `ambos` | `ambos` |
| `--model` | `-m` | Modelo YOLO a usar | `yolov8n.pt` |
| `--max-frames` | `-f` | Límite de frames en video/webcam (0=sin límite) | `0` |

---

## 🔧 Arquitectura del módulo `deteccion_objetos.py`

El script se organizó en funciones separadas para que sea fácil conectarlo, más adelante, a una interfaz web (Flask/FastAPI):

| Función | Descripción |
|---------|-------------|
| `get_frame_generator(source, type)` | Generador único de frames, sirve tanto para imagen, video o webcam |
| `detectar_objetos(model, frame, conf)` | Corre YOLO sobre el frame y devuelve las detecciones con su posición y distancia |
| `guia_usuario(detecciones)` | Arma el texto narrado, con artículos correctos en español |
| `reproducir_audio(texto, archivo)` | Genera el MP3 a partir del texto y lo reproduce con pygame |
| `procesar_imagen(model, path, output)` | Flujo completo para procesar una imagen fija |
| `procesar_video(model, source, type, ...)` | Flujo completo para video/webcam, con resumen al finalizar |

---

## 📝 Notas

- `yolov8n.pt` se descarga automáticamente la primera vez que se corre el sistema.
- El audio queda guardado como `output_audio.mp3` (imágenes) o `resumen_video.mp3` (video/webcam).
- En los modos `webcam` o `video` con `--output mostrar`, se sale de la ventana de video presionando **q**.
- Si falta alguna dependencia al correr el script de consola, se instala automáticamente.
- La detección de GPU es automática; si no hay una disponible, el sistema sigue funcionando en CPU sin ajustes adicionales.
