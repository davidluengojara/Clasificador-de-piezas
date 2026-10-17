# Genera el dataset sintetico de piezas vistas a contraluz
# Clases: arandela, tuerca, tornillo y remache

import os
import random
import numpy as np
import cv2

OUT_DIR = "dataset"
N_TRAIN = 200
N_TEST = 50
SIZE = 96
CLASES = ["arandela", "tuerca", "tornillo", "remache"]


def randu(a, b):
    return a + (b - a) * random.random()


def dibujar_pieza(clase, lienzo=192):
    img = np.zeros((lienzo, lienzo), dtype=np.uint8)
    c = lienzo // 2
    k = randu(0.88, 1.12)

    if clase == "arandela":
        r_ext = int(44 * k)
        r_int = int(randu(18, 22) * k)
        cv2.circle(img, (c, c), r_ext, 255, -1)
        cv2.circle(img, (c, c), r_int, 0, -1)

    elif clase == "tuerca":
        r = 46 * k
        pts = []
        for i in range(6):
            a = np.radians(60 * i)
            pts.append([c + r * np.cos(a), c + r * np.sin(a)])
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 255)
        cv2.circle(img, (c, c), int(randu(19, 23) * k), 0, -1)

    elif clase == "tornillo":
        largo = int(randu(110, 130) * k)
        cabeza = int(randu(14, 18) * k)
        ancho_cab = int(36 * k)
        ancho = int(16 * k)
        x0 = c - largo // 2
        cv2.rectangle(img, (x0, c - ancho_cab // 2), (x0 + cabeza, c + ancho_cab // 2), 255, -1)
        cv2.rectangle(img, (x0 + cabeza, c - ancho // 2), (x0 + largo - 8, c + ancho // 2), 255, -1)
        punta = np.array([[x0 + largo - 8, c - ancho // 2], [x0 + largo, c], [x0 + largo - 8, c + ancho // 2]], dtype=np.int32)
        cv2.fillPoly(img, [punta], 255)
        paso = int(7 * k)
        for x in range(x0 + cabeza + 6, x0 + largo - 12, paso):
            cv2.line(img, (x, c - ancho // 2 - 3), (x + 3, c - ancho // 2), 255, 2)
            cv2.line(img, (x, c + ancho // 2 + 3), (x + 3, c + ancho // 2), 255, 2)

    elif clase == "remache":
        ala = int(randu(5, 7) * k)
        ancho_ala = int(randu(40, 46) * k)
        cuerpo = int(randu(46, 54) * k)
        ancho_cuerpo = int(18 * k)
        vastago = int(randu(50, 60) * k)
        ancho_vas = int(7 * k)
        x0 = c - (ala + cuerpo + vastago) // 2
        cv2.rectangle(img, (x0 + ala, c - ancho_vas // 2), (x0 + ala + cuerpo + vastago, c + ancho_vas // 2), 255, -1)
        cv2.rectangle(img, (x0, c - ancho_ala // 2), (x0 + ala, c + ancho_ala // 2), 255, -1)
        cv2.rectangle(img, (x0 + ala, c - ancho_cuerpo // 2), (x0 + ala + cuerpo, c + ancho_cuerpo // 2), 255, -1)
        cv2.circle(img, (x0 + ala + cuerpo + vastago, c), ancho_vas // 2 + 1, 255, -1)

    return img


def colocar(mascara, severidad):
    h, w = mascara.shape
    ang = randu(0, 360)
    esc = randu(0.92, 1.08)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, esc)
    M[0, 2] += randu(-6, 6) * (1 + severidad)
    M[1, 2] += randu(-6, 6) * (1 + severidad)
    m = cv2.warpAffine(mascara, M, (w, h), flags=cv2.INTER_LINEAR)
    m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return m.astype(np.float32) / 255


def contraluz(m, severidad):
    fondo = randu(190, 240)
    pieza = randu(20, 60)
    # iluminacion un poco irregular del panel de contraluz
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    grad = randu(-25, 25) * xx + randu(-25, 25) * yy
    img = fondo + grad
    img = img * (1 - m) + pieza * m
    sigma = randu(0, 0.8 + 1.5 * severidad)
    if sigma > 0.1:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = img + np.random.normal(0, 4 + 10 * severidad, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def generar(split, n, severidad):
    for clase in CLASES:
        carpeta = os.path.join(OUT_DIR, split, clase)
        os.makedirs(carpeta, exist_ok=True)
        for i in range(n):
            m = colocar(dibujar_pieza(clase), severidad)
            img = contraluz(m, severidad)
            cv2.imwrite(os.path.join(carpeta, clase + "_" + str(i).zfill(4) + ".png"), img)


def generar_escena():
    # tramo de cinta con varias piezas, dos de ellas tocandose
    ancho, alto = 640, 400
    mascara = np.zeros((alto, ancho), dtype=np.float32)
    piezas = [("tuerca", 80, 90), ("tornillo", 230, 80), ("arandela", 400, 95), ("remache", 550, 90),
              ("remache", 90, 260), ("arandela", 230, 300), ("tornillo", 420, 290), ("tuerca", 560, 300),
              ("arandela", 300, 200), ("tuerca", 342, 202)]
    for clase, x, y in piezas:
        m = colocar(dibujar_pieza(clase), 0.0)
        y0 = y - SIZE // 2
        x0 = x - SIZE // 2
        mascara[y0:y0 + SIZE, x0:x0 + SIZE] = np.maximum(mascara[y0:y0 + SIZE, x0:x0 + SIZE], m)
    yy, xx = np.mgrid[0:alto, 0:ancho]
    img = 225 - 20 * xx / ancho - 10 * yy / alto
    img = img * (1 - mascara) + 40 * mascara
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img = img + np.random.normal(0, 6, img.shape)
    cv2.imwrite("escena.png", np.clip(img, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    generar("train", N_TRAIN, severidad=0.3)
    generar("test", N_TEST, severidad=0.5)
    generar_escena()
    print("Hecho:", OUT_DIR)
