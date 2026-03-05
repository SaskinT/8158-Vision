"""
AprilTag 36h11 - Kamera + Uzaklık + Açı Hesaplama
====================================================
Kurulum:
    pip install opencv-python pupil-apriltags numpy

Çalıştırma:
    python apriltag_kamera.py

Açı Tanımları:
    Yaw   (Yatay açı)   : Tag'in kameraya göre sağ/sol açısı
    Pitch (Dikey açı)   : Tag'in kameraya göre yukarı/aşağı açısı
    Roll  (Dönme açısı) : Tag'in kendi ekseni etrafındaki dönüşü
"""

import cv2
import numpy as np
import sys
import time

try:
    from pupil_apriltags import Detector
except ImportError:
    print("HATA: Şu komutu çalıştır -->  pip install pupil-apriltags")
    sys.exit(1)


# ══════════════════════════════════════════════════════
#  AYARLAR
# ══════════════════════════════════════════════════════
KAMERA_ID   = 0
GENISLIK    = 1280
YUKSEKLIK   = 720
DECIMATION  = 2.0
TAG_BOYUTU  = 0.055    # metre (siyah karenin dış kenarı)
KAMERA_FOV  = 65.0    # derece
# ══════════════════════════════════════════════════════


# Kamera iç parametreleri
f_px = (GENISLIK / 2.0) / np.tan(np.radians(KAMERA_FOV / 2.0))
kamera_matrisi = np.array([
    [f_px,   0,  GENISLIK  / 2.0],
    [0,    f_px, YUKSEKLIK / 2.0],
    [0,       0,             1.0]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))


# Dedektör
detector = Detector(
    families="tag36h11",
    nthreads=2,
    quad_decimate=DECIMATION,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
)

fps = 0
timerStart = False
startTime = 0


# ── Pose hesaplama ────────────────────────────────────────────────────────────
def pose_hesapla(tag):
    """
    PnP ile tag'in 3D konumunu ve açısını hesaplar.
    Döner: uzaklık (m), yaw (°), pitch (°), roll (°), rvec, tvec
    """
    yari = TAG_BOYUTU / 2.0
    obj_pts = np.array([
        [-yari,  yari, 0],
        [ yari,  yari, 0],
        [ yari, -yari, 0],
        [-yari, -yari, 0],
    ], dtype=np.float64)

    basari, rvec, tvec = cv2.solvePnP(
        obj_pts, tag.corners.astype(np.float64),
        kamera_matrisi, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not basari:
        return None, None, None, None, None, None

    # Uzaklık
    uzaklik = float(np.linalg.norm(tvec))

    # Rotation matrix → Euler açıları
    R, _ = cv2.Rodrigues(rvec)

    # Yaw (sağ/sol), Pitch (yukarı/aşağı), Roll (dönüş)
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))

    # Kameraya göre yatay ve dikey açı (merkez noktasından)
    tx, ty, tz = tvec[0][0], tvec[1][0], tvec[2][0]
    aci_yatay  = np.degrees(np.arctan2(tx, tz))   # sağ + / sol -
    aci_dikey  = np.degrees(np.arctan2(-ty, tz))  # yukarı + / aşağı -

    return uzaklik, yaw, pitch, roll, aci_yatay, aci_dikey, rvec, tvec


# ── Yardımcı ─────────────────────────────────────────────────────────────────
def renk_uzakliga_gore(m):
    if m < 0.5:   return (0, 60, 255)
    elif m < 1.5: return (0, 200, 255)
    else:         return (0, 255, 80)


def aci_cubugu_ciz(frame, x, y, genislik, aci, max_aci, etiket, renk):
    """Küçük açı göstergesi çizer."""
    yukseklik = 10
    dolu = int(abs(aci) / max_aci * genislik / 2)
    dolu = min(dolu, genislik // 2)
    orta = x + genislik // 2

    # Arka plan
    cv2.rectangle(frame, (x, y), (x + genislik, y + yukseklik), (40, 40, 40), -1)
    # Orta çizgi
    cv2.line(frame, (orta, y), (orta, y + yukseklik), (100, 100, 100), 1)
    # Dolgu (sağa veya sola)
    if aci >= 0:
        cv2.rectangle(frame, (orta, y), (orta + dolu, y + yukseklik), renk, -1)
    else:
        cv2.rectangle(frame, (orta - dolu, y), (orta, y + yukseklik), renk, -1)
    # Etiket
    cv2.putText(frame, f"{etiket}: {aci:+.1f}°",
                (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, renk, 1)


def bilgi_kutusu_ciz(frame, x, y, uzaklik, yaw, pitch, roll, aci_yatay, aci_dikey, renk):
    """Tag üzerinde yarı şeffaf bilgi paneli çizer."""
    satir_y = [y, y+20, y+40, y+60, y+80, y+100]

    ov = frame.copy()
    cv2.rectangle(ov, (x - 6, y - 16), (x + 175, y + 110), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    cm = uzaklik * 100
    uzaklik_yazi = f"{cm:.1f} cm" if cm < 100 else f"{uzaklik:.2f} m"

    satirlar = [
        (f"Uzaklik : {uzaklik_yazi}",       renk),
        (f"Yatay   : {aci_yatay:+.1f} deg", (100, 220, 255)),
        (f"Dikey   : {aci_dikey:+.1f} deg", (255, 200, 100)),
        (f"Yaw     : {yaw:+.1f} deg",       (180, 255, 180)),
        (f"Pitch   : {pitch:+.1f} deg",     (255, 180, 255)),
        (f"Roll    : {roll:+.1f} deg",      (255, 255, 150)),
    ]
    for i, (metin, r) in enumerate(satirlar):
        cv2.putText(frame, metin, (x, satir_y[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, r, 1)


# ── Kamerayı Aç ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(KAMERA_ID)
if not cap.isOpened():
    print(f"HATA: Kamera {KAMERA_ID} açılamadı.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  GENISLIK)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, YUKSEKLIK)
print("Kamera acildi! Cikmak icin 'q' bas.\n")

onceki_ids = set()

# ── Ana Döngü ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame.any():
        fps +=1
        if timerStart:
            startTime = time.time()

    gri  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gri)

    # Terminal (değişince)
    simdi_ids = {t.tag_id for t in tags}
    if simdi_ids != onceki_ids:
        print(f"{len(tags)} tag:" if tags else "  [Tag yok]")
        onceki_ids = simdi_ids

    # Merkez yatay/dikey referans çizgileri
    cv2.line(frame, (GENISLIK//2, 0), (GENISLIK//2, YUKSEKLIK), (60, 60, 60), 1)
    cv2.line(frame, (0, YUKSEKLIK//2), (GENISLIK, YUKSEKLIK//2), (60, 60, 60), 1)

    for tag in tags:
        sonuc = pose_hesapla(tag)
        if sonuc[0] is None:
            continue
        uzaklik, yaw, pitch, roll, aci_yatay, aci_dikey, rvec, tvec = sonuc

        koseler = tag.corners.astype(int)
        merkez  = tag.center.astype(int)
        renk    = renk_uzakliga_gore(uzaklik)

        # Çerçeve
        for i in range(4):
            cv2.line(frame, tuple(koseler[i]), tuple(koseler[(i+1) % 4]), renk, 2)
        for j, k in enumerate(koseler):
            cv2.circle(frame, tuple(k), 5,
                       (0, 0, 255) if j == 0 else (180, 80, 0), -1)
        cv2.circle(frame, tuple(merkez), 4, renk, -1)

        # 3D eksen
        cv2.drawFrameAxes(frame, kamera_matrisi, dist_coeffs,
                          rvec, tvec, TAG_BOYUTU * 0.5)

        # Kameradan tag merkezine çizgi
        cam_merkez = (GENISLIK // 2, YUKSEKLIK // 2)
        cv2.line(frame, cam_merkez, tuple(merkez), (80, 80, 80), 1)

        # Açı çubuğu göstergeleri (sol alt)
        panel_x, panel_y = 20, YUKSEKLIK - 80
        aci_cubugu_ciz(frame, panel_x, panel_y,      200, aci_yatay, 45,
                       "Yatay ", (100, 220, 255))
        aci_cubugu_ciz(frame, panel_x, panel_y + 30, 200, aci_dikey, 45,
                       "Dikey ", (255, 200, 100))

        # Bilgi kutusu (tag yanında)
        bx = min(merkez[0] + 20, GENISLIK - 185)
        by = max(merkez[1] - 20, 20)
        bilgi_kutusu_ciz(frame, bx, by, uzaklik, yaw, pitch, roll,
                         aci_yatay, aci_dikey, renk)

        # Terminal
        cm = uzaklik * 100
        print(f"  ID={tag.tag_id} | "
              f"Uzaklik={'%5.1fcm'%cm if cm<100 else '%.2fm'%uzaklik} | "
              f"Yatay={aci_yatay:+6.1f}° | Dikey={aci_dikey:+6.1f}° | "
              f"Yaw={yaw:+6.1f}° | Pitch={pitch:+6.1f}° | Roll={roll:+6.1f}°")

    # Üst bilgi bandı
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, 0), (GENISLIK, 38), (0, 0, 0), -1)
    cv2.addWeighted(ov2, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame,
                f"AprilTag 36h11  |  Tag: {len(tags)}  |  "
                f"Boyut: {TAG_BOYUTU*100:.0f}cm  |  Q=cikis | Frames : {int(fps)} | Time: {time.time - startTime}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("AprilTag - Uzaklik & Aci", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nKapatildi.")