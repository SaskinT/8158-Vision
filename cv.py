"""
AprilTag 36h11 - Kamera + Uzaklık Ölçümü
=========================================
"""

import cv2
import numpy as np
import sys

try:
    from pupil_apriltags import Detector
except ImportError:
    print("HATA: Şu komutu çalıştır -->  pip install pupil-apriltags")
    sys.exit(1)


# ══════════════════════════════════════════════════════
#  AYARLAR — Bunları kendi durumuna göre değiştir
# ══════════════════════════════════════════════════════
KAMERA_ID   = 0       # Kamera numarası (0 = dahili, 1 = USB)
GENISLIK    = 1280    # Çözünürlük genişliği
YUKSEKLIK   = 720     # Çözünürlük yüksekliği
DECIMATION  = 2.0     # 1.0 = hassas&yavaş | 2.0 = hızlı (önerilen)

# Tag fiziksel boyutu (metre cinsinden — siyah karenin dış kenarı)
# Örnek: 15 cm'lik tag → 0.15
TAG_BOYUTU  = 0.15    # metre

# Kamera yatay görüş açısı (derece) — çoğu webcam ~60-70 derece
KAMERA_FOV  = 65.0    # derece
# ══════════════════════════════════════════════════════


# Odak uzaklığını FOV'dan tahmin et (piksel cinsinden)
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


def uzaklik_hesapla(tag):
    """Tag köşelerinden PnP çözümü ile uzaklık hesaplar (metre)."""
    yari = TAG_BOYUTU / 2.0
    obj_pts = np.array([
        [-yari,  yari, 0],
        [ yari,  yari, 0],
        [ yari, -yari, 0],
        [-yari, -yari, 0],
    ], dtype=np.float64)

    img_pts = tag.corners.astype(np.float64)

    basari, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, kamera_matrisi, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not basari:
        return None, None, None

    return float(np.linalg.norm(tvec)), rvec, tvec


def renk_uzakliga_gore(metre):
    """Yakın = kırmızı, orta = sarı, uzak = yeşil."""
    if metre < 0.5:
        return (0, 60, 255)
    elif metre < 1.5:
        return (0, 200, 255)
    else:
        return (0, 255, 80)


# Kamerayı aç
cap = cv2.VideoCapture(KAMERA_ID)
if not cap.isOpened():
    print(f"HATA: Kamera {KAMERA_ID} açılamadı. KAMERA_ID'yi değiştir.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  GENISLIK)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, YUKSEKLIK)

print("Kamera acildi! Cikmak icin 'q' tusuna bas.\n")

onceki_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("HATA: Goruntu alinamadi.")
        break

    gri  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gri)

    # Terminal çıktısı (sadece değişince)
    simdi_ids = {t.tag_id for t in tags}
    if simdi_ids != onceki_ids:
        onceki_ids = simdi_ids

    for tag in tags:
        uzaklik, rvec, tvec = uzaklik_hesapla(tag)
        koseler = tag.corners.astype(int)
        merkez  = tag.center.astype(int)
        renk    = renk_uzakliga_gore(uzaklik) if uzaklik else (0, 255, 0)

        # Çerçeve çiz
        for i in range(4):
            cv2.line(frame, tuple(koseler[i]), tuple(koseler[(i+1) % 4]), renk, 2)

        # Köşe noktaları (0. köşe kırmızı)
        for j, k in enumerate(koseler):
            cv2.circle(frame, tuple(k), 5,
                       (0, 0, 255) if j == 0 else (200, 100, 0), -1)

        # Merkez noktası
        cv2.circle(frame, tuple(merkez), 4, renk, -1)

        # 3D eksen çizgisi
        if rvec is not None:
            cv2.drawFrameAxes(frame, kamera_matrisi, dist_coeffs,
                              rvec, tvec, TAG_BOYUTU * 0.5)

        # Bilgi kutusu
        if uzaklik is not None:
            cm = uzaklik * 100
            uzaklik_yazi = f"{cm:.1f} cm" if cm < 100 else f"{uzaklik:.2f} m"

            x, y = merkez[0] - 60, merkez[1] - 55
            overlay = frame.copy()
            cv2.rectangle(overlay, (x - 4, y - 22), (x + 140, y + 32), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            cv2.putText(frame, f"ID: {tag.tag_id}",
                        (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, f"Uzaklik: {uzaklik_yazi}",
                        (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)

            print(f"  ID={tag.tag_id}  |  Uzaklik={uzaklik_yazi}  |  "
                  f"Merkez=({merkez[0]}, {merkez[1]})")

    # Üst bilgi bandı
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, 0), (GENISLIK, 38), (0, 0, 0), -1)
    cv2.addWeighted(ov2, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame,
                f"AprilTag 36h11  |  Tag: {len(tags)}  |  "
                f"Tag boyutu: {TAG_BOYUTU*100:.0f}cm  |  Q=cikis",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("AprilTag - Uzaklik Olcumu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nKapatildi.")