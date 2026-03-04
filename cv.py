"""
AprilTag 36h11 - Kamera + Uzaklık + Açı Ölçümü
================================================
Yeni özellik: Tag'in kameraya göre Yaw, Pitch, Roll açıları (derece)
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
KAMERA_ID   = 2       # Kamera numarası (0 = dahili, 1 = USB)
GENISLIK    = 1280    # Çözünürlük genişliği
YUKSEKLIK   = 720     # Çözünürlük yüksekliği
DECIMATION  = 2.0     # 1.0 = hassas&yavaş | 2.0 = hızlı (önerilen)

# Tag fiziksel boyutu (metre cinsinden — siyah karenin dış kenarı)
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


def aci_hesapla(rvec):
    """
    Rotation vektöründen (rvec) Euler açılarını hesaplar.

    Kamera koordinat sistemi:
      - X : sağa
      - Y : aşağıya
      - Z : kameradan uzağa (derinlik)

    Döndürülen açılar (derece):
      - yaw   : tag'in soldan-sağa dönüş açısı  (Y ekseni etrafında)
      - pitch : tag'in yukarı-aşağı tilt açısı  (X ekseni etrafında)
      - roll  : tag'in kendi düzleminde dönüşü   (Z ekseni etrafında)

    Açı = 0° → tag kameraya tam dik bakıyor
    """
    # Rodrigues → 3x3 Rotasyon matrisi
    R, _ = cv2.Rodrigues(rvec)

    # ZYX (yaw-pitch-roll) Euler açıları
    # R = Rz * Ry * Rx  şeklinde ayrıştırılır
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    tekil = sy < 1e-6  # gimbal lock kontrolü

    if not tekil:
        roll  = np.degrees(np.arctan2( R[2, 1],  R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0],  sy))
        yaw   = np.degrees(np.arctan2( R[1, 0],  R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2],  R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0],  sy))
        yaw   = 0.0

    return yaw, pitch, roll


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
print(f"{'ID':>4}  {'Uzaklik':>10}  {'Yaw':>8}  {'Pitch':>8}  {'Roll':>8}")
print("-" * 50)

onceki_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("HATA: Goruntu alinamadi.")
        break

    gri  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gri)

    simdi_ids = {t.tag_id for t in tags}
    if simdi_ids != onceki_ids:
        onceki_ids = simdi_ids

    for tag in tags:
        uzaklik, rvec, tvec = uzaklik_hesapla(tag)
        koseler = tag.corners.astype(int)
        merkez  = tag.center.astype(int)
        renk    = renk_uzakliga_gore(uzaklik) if uzaklik else (0, 255, 0)

        # Açıları hesapla
        yaw, pitch, roll = (0.0, 0.0, 0.0)
        if rvec is not None:
            yaw, pitch, roll = aci_hesapla(rvec)

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

        # ── Bilgi kutusu (uzaklık + açılar) ──────────────────────────
        if uzaklik is not None:
            cm = uzaklik * 100
            uzaklik_yazi = f"{cm:.1f} cm" if cm < 100 else f"{uzaklik:.2f} m"

            x, y = merkez[0] - 70, merkez[1] - 80

            # Arkaplan kutusu (4 satır için daha uzun)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x - 4, y - 22), (x + 185, y + 76), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            # Satır 1 — ID
            cv2.putText(frame, f"ID: {tag.tag_id}",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # Satır 2 — Uzaklık
            cv2.putText(frame, f"Uzaklik: {uzaklik_yazi}",
                        (x, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)

            # Satır 3 — Yaw & Pitch
            cv2.putText(frame, f"Yaw:{yaw:+.1f}  Pitch:{pitch:+.1f}",
                        (x, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2)

            # Satır 4 — Roll
            cv2.putText(frame, f"Roll:{roll:+.1f} deg",
                        (x, y + 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2)

            # Terminal çıktısı
            print(f"  ID={tag.tag_id:>3}  |  {uzaklik_yazi:>8}  |  "
                  f"Yaw={yaw:+7.1f}°  Pitch={pitch:+7.1f}°  Roll={roll:+7.1f}°")

    # Üst bilgi bandı
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, 0), (GENISLIK, 38), (0, 0, 0), -1)
    cv2.addWeighted(ov2, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame,
                f"AprilTag 36h11  |  Tag: {len(tags)}  |  "
                f"Tag boyutu: {TAG_BOYUTU*100:.0f}cm  |  Q=cikis",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("AprilTag - Uzaklik & Aci Olcumu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nKapatildi.")