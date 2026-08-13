"""
Dependencias (instalar con):
    pip install ultralytics opencv-python-headless pillow matplotlib gtts pygame

Uso:
    python deteccion_objetos.py --source imagen.jpg --type image
    python deteccion_objetos.py --source video.mp4 --type video
    python deteccion_objetos.py --source 0 --type webcam

Modes de salida:
    --output narrar    -> genera audio con guia de voz (default)
    --output mostrar   -> muestra imagen/frames con bounding boxes
    --output ambos     -> narrar y mostrar
"""

# ─────────────────────────────────────────────────────────
# INSTALACIONES
# ─────────────────────────────────────────────────────────
import subprocess
import sys

REQUIRED_PACKAGES = [
    "ultralytics",
    "opencv-python-headless",
    "pillow",
    "matplotlib",
    "gtts",
    "pygame",
]

def install_if_missing(package):
    try:
        __import__(package.replace("-", "_").split("-headless")[0])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in REQUIRED_PACKAGES:
    install_if_missing(pkg)


# ─────────────────────────────────────────────────────────
# IMPORTACIONES
# ─────────────────────────────────────────────────────────
import os
import sys
import argparse
import base64
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import pygame


# ─────────────────────────────────────────────────────────
# CONFIGURACION GLOBAL
# ─────────────────────────────────────────────────────────

# Modelo YOLO
MODEL_PATH = "yolov8n.pt"

# Diccionario de traduccion
TRADUCCION = {
    'person': 'persona', 'bicycle': 'bicicleta', 'car': 'auto',
    'motorcycle': 'motocicleta', 'airplane': 'avion', 'bus': 'autobus',
    'train': 'tren', 'truck': 'camion', 'boat': 'barco',
    'traffic light': 'semaforo', 'fire hydrant': 'hidrante',
    'stop sign': 'senal de alto', 'parking meter': 'parquimetro',
    'bench': 'banco', 'bird': 'pajaro', 'cat': 'gato', 'dog': 'perro',
    'horse': 'caballo', 'sheep': 'oveja', 'cow': 'vaca',
    'elephant': 'elefante', 'bear': 'oso', 'zebra': 'cebra',
    'giraffe': 'jirafa', 'backpack': 'mochila', 'umbrella': 'paraguas',
    'handbag': 'bolso', 'tie': 'corbata', 'suitcase': 'maleta',
    'frisbee': 'frisbee', 'skis': 'esquies', 'snowboard': 'snowboard',
    'sports ball': 'pelota', 'kite': 'cometa',
    'baseball bat': 'bate de beisbol', 'baseball glove': 'guante de beisbol',
    'skateboard': 'patineta', 'surfboard': 'tabla de surf',
    'tennis racket': 'raqueta de tenis', 'bottle': 'botella',
    'wine glass': 'copa de vino', 'cup': 'taza', 'fork': 'tenedor',
    'knife': 'cuchillo', 'spoon': 'cuchara', 'bowl': 'tazon',
    'banana': 'platano', 'apple': 'manzana', 'sandwich': 'sandwich',
    'orange': 'naranja', 'broccoli': 'brocoli', 'carrot': 'zanahoria',
    'hot dog': 'hot dog', 'pizza': 'pizza', 'donut': 'dona',
    'cake': 'pastel', 'chair': 'silla', 'couch': 'sofa',
    'potted plant': 'planta', 'bed': 'cama',
    'dining table': 'mesa de comedor', 'toilet': 'bano', 'tv': 'tv',
    'laptop': 'laptop', 'mouse': 'raton', 'remote': 'control remoto',
    'keyboard': 'teclado', 'cell phone': 'celular',
    'microwave': 'microondas', 'oven': 'horno', 'toaster': 'tostadora',
    'sink': 'fregadero', 'refrigerator': 'refrigerador', 'book': 'libro',
    'clock': 'reloj', 'vase': 'florero', 'scissors': 'tijeras',
    'teddy bear': 'oso de peluche', 'hair drier': 'secador de pelo',
    'toothbrush': 'cepillo de dientes',
}


# ─────────────────────────────────────────────────────────
# CAPTURA DE FUENTES (imagen / video / webcam)
# ─────────────────────────────────────────────────────────

def read_image(image_path: str) -> np.ndarray:
    """Lee una imagen y la convierte a RGB."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo abrir la imagen: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_video(video_path: str) -> cv2.VideoCapture:
    """Abre un archivo de video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    return cap


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    """Abre la camara web."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir la camara indice: {camera_index}")
    return cap


def frame_generator(cap: cv2.VideoCapture):
    """Generador de frames desde VideoCapture (video o webcam)."""
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def single_frame(image: np.ndarray):
    """Generador de un unico frame (imagen estatica)."""
    yield image


def get_frame_generator(source=None, source_type: str = "image"):
    """
    Generador unificado de frames.

    Args:
        source: ruta de imagen/video o indice de camara
        source_type: 'image' | 'video' | 'webcam'
    """
    if source_type == "image":
        frame = read_image(str(source))
        yield from single_frame(frame)
    elif source_type == "video":
        cap = read_video(str(source))
        yield from frame_generator(cap)
    elif source_type == "webcam":
        idx = 0 if source is None else int(source)
        cap = open_webcam(idx)
        yield from frame_generator(cap)
    else:
        raise ValueError(f"Tipo de fuente desconocido: {source_type}")


# ─────────────────────────────────────────────────────────
# LOGICA DE DETECCION
# ─────────────────────────────────────────────────────────

def obtener_posicion(x_center: float, image_width: int) -> str:
    """Devuelve la posicion horizontal del objeto en la imagen."""
    tercio_izquierdo = image_width / 3
    tercio_derecho = 2 * image_width / 3

    if x_center < tercio_izquierdo:
        return "a la izquierda"
    elif x_center < tercio_derecho:
        return "al frente"
    else:
        return "a la derecha"


def estimar_distancia(x1, y1, x2, y2, frame_width, frame_height) -> str:
    """Estima la distancia relativa basandose en el area del bounding box."""
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    proporcion = box_area / frame_area

    if proporcion >= 0.20:
        return "muy cerca"
    elif proporcion >= 0.08:
        return "cerca"
    elif proporcion >= 0.03:
        return "a media distancia"
    else:
        return "lejos"


def detectar_objetos(model: YOLO, frame: np.ndarray, conf: float = 0.35) -> tuple:
    """
    Detecta objetos en un frame y devuelve lista de detecciones y resultado YOLO.

    Returns:
        (detecciones, result) donde detecciones es lista de dicts con
        keys: label_en, label_es, confidence, bbox, position, distance
    """
    results = model(frame, verbose=False, conf=conf)
    result = results[0]

    image_width = frame.shape[1]
    image_height = frame.shape[0]
    detecciones = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        label = result.names[class_id]
        label_es = TRADUCCION.get(label, label)
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2

        posicion = obtener_posicion(x_center, image_width)
        distancia = estimar_distancia(x1, y1, x2, y2, image_width, image_height)

        detecciones.append({
            "label_en": label,
            "label_es": label_es,
            "confidence": round(conf, 2),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "position": posicion,
            "distance": distancia,
        })

    return detecciones, result


# ─────────────────────────────────────────────────────────
# GENERACION DE MENSAJE DE GUIA
# ─────────────────────────────────────────────────────────

def guia_usuario(detecciones: list) -> str:
    """
    Genera un mensaje de guia a partir de la lista de detecciones.

    Prioriza obstaculos muy cercanos al frente e incluye
    articulos gramaticales correctos en espanol.
    """
    if not detecciones:
        return "Camino despejado. Puede avanzar."

    conteo_objetos = Counter([
        (d.get("label_es", "objeto"), d.get("position", "centro"))
        for d in detecciones
    ])

    frases_agrupadas = []
    bloqueo_frente = False

    for (obj, pos), cantidad in conteo_objetos.items():
        pos_natural = (
            pos.replace("al frente", "justo al frente")
               .replace("a la", "a tu")
        )

        peligro = any(
            d.get("distance", "media") == "muy cerca"
            and d.get("position", "") == "al frente"
            for d in detecciones
            if d.get("label_es") == obj
        )

        if peligro:
            bloqueo_frente = True

        if cantidad > 1:
            ultima = obj.lower()[-1]
            plural = obj + "s" if ultima in "aeiouaeiou" else obj + "es"
            frases_agrupadas.append(f"{cantidad} {plural} {pos_natural}")
        else:
            articulo = "una" if obj.lower().endswith("a") else "un"
            frases_agrupadas.append(f"{articulo} {obj} {pos_natural}")

    resumen = ", ".join(frases_agrupadas)

    if bloqueo_frente:
        return f"Atencion. Detecto {resumen}. Hay un obstaculo muy cerca al frente. Detente."
    return f"Hay {resumen}"


# ─────────────────────────────────────────────────────────
# AUDIO (gTTS + pygame)
# ─────────────────────────────────────────────────────────

def reproducir_audio(texto: str, archivo: str = "output_audio.mp3") -> None:
    """Genera y reproduce audio con el texto dado."""
    tts = gTTS(text=texto, lang="es", tld="com.mx")
    tts.save(archivo)

    pygame.mixer.init()
    pygame.mixer.music.load(archivo)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()


# ─────────────────────────────────────────────────────────
# MODO IMAGEN
# ─────────────────────────────────────────────────────────

def procesar_imagen(model: YOLO, image_path: str, output_mode: str = "ambos") -> None:
    """Procesa una imagen estatica."""
    print(f"\n[INFO] Procesando imagen: {image_path}")
    frame = read_image(image_path)
    detecciones, result = detectar_objetos(model, frame)

    # Imprimir detecciones en consola
    print(f"\n{'='*50}")
    print(f"  Objetos detectados: {len(detecciones)}")
    print(f"{'='*50}")
    for d in detecciones:
        print(
            f"  * {d['label_es']:20s} | {d['position']:15s} | "
            f"{d['distance']:15s} | conf: {d['confidence']}"
        )

    texto_guia = guia_usuario(detecciones)
    print(f"\n[GUIA] {texto_guia}\n")

    if output_mode in ("mostrar", "ambos"):
        annotated = result.plot()
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated)
        plt.axis("off")
        plt.title("Deteccion de Objetos")
        plt.tight_layout()
        plt.show()

    if output_mode in ("narrar", "ambos"):
        reproducir_audio(texto_guia)


# ─────────────────────────────────────────────────────────
# MODO VIDEO / WEBCAM
# ─────────────────────────────────────────────────────────

def procesar_video(
    model: YOLO,
    source,
    source_type: str,
    output_mode: str = "ambos",
    max_frames: int = 0,
) -> None:
    """
    Procesa un video o stream de webcam.

    Args:
        max_frames: 0 = procesar todo; >0 = limitar numero de frames
    """
    print(f"\n[INFO] Procesando {source_type}: {source}")
    print("[INFO] Presiona 'q' en la ventana de video para salir.\n")

    todas_detecciones = []

    for i, frame in enumerate(get_frame_generator(source, source_type)):
        detecciones, result = detectar_objetos(model, frame)
        todas_detecciones.extend(detecciones)

        if output_mode in ("mostrar", "ambos"):
            annotated = result.plot()
            # Convertir de RGB a BGR para cv2
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imshow("Deteccion de Objetos", annotated_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if max_frames > 0 and i >= max_frames - 1:
            break

    cv2.destroyAllWindows()

    # ── Narrar resumen al final ──────────────────────────
    # Consolidar detecciones unicas por (objeto, posicion)
    objetos_unicos = {}
    for d in todas_detecciones:
        key = (d.get("label_es", "objeto"), d.get("position", "centro"))
        if key not in objetos_unicos:
            objetos_unicos[key] = d

    objetos_por_tipo = defaultdict(list)
    for d in objetos_unicos.values():
        objetos_por_tipo[d["label_es"]].append(d["position"])

    frases_agrupadas = []
    for obj, posiciones in objetos_por_tipo.items():
        posiciones = list(dict.fromkeys(posiciones))
        if len(posiciones) == 1:
            articulo = "una" if obj.lower().endswith("a") else "un"
            frases_agrupadas.append(f"{articulo} {obj} {posiciones[0]}")
        else:
            ultima = obj.lower()[-1]
            plural = obj + "s" if ultima in "aeiouaeiou" else obj + "es"
            frases_agrupadas.append(
                f"{len(posiciones)} {plural}, " + " y ".join(posiciones)
            )

    texto_final = ", ".join(frases_agrupadas) if frases_agrupadas else "Camino despejado."
    resumen = f"Resumen: Hay {texto_final}." if frases_agrupadas else texto_final

    print(f"\n[RESUMEN] {resumen}\n")

    if output_mode in ("narrar", "ambos"):
        reproducir_audio(resumen, "resumen_video.mp3")


# ─────────────────────────────────────────────────────────
# Pruebas
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Deteccion de Objetos con Guia por Voz"
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Ruta de imagen/video o indice de camara (default: None -> webcam 0)"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["image", "video", "webcam"],
        default="webcam",
        help="Tipo de fuente: image | video | webcam (default: webcam)"
    )
    parser.add_argument(
        "--output", "-o",
        choices=["narrar", "mostrar", "ambos"],
        default="ambos",
        help="Modo de salida: narrar | mostrar | ambos (default: ambos)"
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL_PATH,
        help=f"Ruta o nombre del modelo YOLO (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--max-frames", "-f",
        type=int,
        default=0,
        help="Maximo de frames a procesar en video/webcam (0=todos)"
    )

    args = parser.parse_args()

    # Cargar modelo
    print(f"[INFO] Cargando modelo: {args.model}")
    model = YOLO(args.model)

    # Procesar segun tipo
    if args.type == "image":
        if args.source is None:
            parser.error("--source es obligatorio para --type image")
        procesar_imagen(model, args.source, args.output)
    else:
        source = args.source if args.source is not None else 0
        procesar_video(model, source, args.type, args.output, args.max_frames)


if __name__ == "__main__":
    main()
