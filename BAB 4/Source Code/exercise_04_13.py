
import os

# Harus diatur sebelum TensorFlow diimpor
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import cv2
import numpy as np

# ==========================================
# 1. BUKA KAMERA TERLEBIH DAHULU
# ==========================================

print("Membuka kamera...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Jika kamera pertama gagal, coba indeks berikutnya
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    print("Periksa izin kamera dan apakah kamera digunakan aplikasi lain.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Kamera berhasil dibuka!")

# ==========================================
# 2. TAMPILKAN KAMERA SEBELUM MEMUAT MODEL
# ==========================================

print("Menampilkan kamera. Tekan Q untuk keluar.")

cv2.namedWindow("Image Classification Camera", cv2.WINDOW_NORMAL)

# ==========================================
# 3. MUAT MODEL MOBILE NET V2
# ==========================================

model = None
preprocess_input = None
decode_predictions = None

try:
    print("Memuat TensorFlow dan MobileNetV2...")

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input as mobilenet_preprocess
    )
    from tensorflow.keras.applications.imagenet_utils import (
        decode_predictions as decode
    )

    preprocess_input = mobilenet_preprocess
    decode_predictions = decode

    print("Memuat bobot ImageNet. Tunggu jika baru pertama kali...")

    model = MobileNetV2(
        weights="imagenet",
        include_top=True
    )

    print("Model berhasil dimuat!")

except Exception as e:
    print("\nGAGAL MEMUAT MODEL:")
    print(e)
    print("Kamera tetap berjalan tanpa klasifikasi.")
    print("Periksa koneksi internet dan instalasi TensorFlow.")

# ==========================================
# 4. PENGATURAN PREDIKSI
# ==========================================

last_prediction_time = 0
prediction_interval = 1.5

prediction_text = "Menunggu prediksi..."
confidence = 0.0

# ==========================================
# 5. LOOP KAMERA
# ==========================================

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        # Membalik gambar seperti cermin
        frame = cv2.flip(frame, 1)

        current_time = time.time()

        # Jalankan prediksi setiap 1,5 detik
        if (
            model is not None
            and current_time - last_prediction_time >= prediction_interval
        ):

            last_prediction_time = current_time

            try:
                # OpenCV menggunakan BGR, model membutuhkan RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MobileNetV2 membutuhkan gambar 224 x 224
                resized = cv2.resize(rgb, (224, 224))

                # Ubah menjadi array float
                image = resized.astype(np.float32)

                # Tambahkan dimensi batch
                image = np.expand_dims(image, axis=0)

                # Preprocessing khusus MobileNetV2
                image = preprocess_input(image)

                # Prediksi
                preds = model.predict(image, verbose=0)

                # Ambil hasil dengan probabilitas tertinggi
                result = decode_predictions(preds, top=1)[0][0]

                prediction_text = result[1]
                confidence = float(result[2])

                print(
                    f"Prediksi: {prediction_text} "
                    f"| Confidence: {confidence:.2%}"
                )

            except Exception as e:
                prediction_text = "Prediksi gagal"
                print("Error prediksi:", e)

        # ======================================
        # 6. TAMPILKAN HASIL PADA KAMERA
        # ======================================

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 90),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "IMAGE CLASSIFICATION - MobileNetV2",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Object: {prediction_text}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2%}",
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1
        )

        cv2.imshow("Image Classification Camera", frame)

        # Tekan Q atau ESC untuk keluar
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Kamera ditutup.")
