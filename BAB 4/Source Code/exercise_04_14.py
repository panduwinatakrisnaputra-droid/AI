
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import time
import numpy as np

# ==========================================
# 1. PILIH MODEL
# ==========================================

print("Pilih model klasifikasi:")
print("1. VGG16")
print("2. ResNet50")
print("3. MobileNetV2")
print("4. DenseNet201")
print("5. InceptionV3")

choice = input("Masukkan pilihan (1-5): ").strip()

models = {
    "1": ("vgg16", 224),
    "2": ("resnet50", 224),
    "3": ("mobilenetv2", 224),
    "4": ("densenet201", 224),
    "5": ("inceptionv3", 299)
}

if choice not in models:
    print("Pilihan tidak valid. Menggunakan MobileNetV2.")
    choice = "3"

name, sz = models[choice]

# ==========================================
# 2. BUKA KAMERA TERLEBIH DAHULU
# ==========================================

print("\nMembuka kamera...")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    print("Kamera indeks 0 gagal. Mencoba indeks 1...")
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Kamera tidak bisa dibuka.")
    print("Periksa izin kamera Windows dan tutup aplikasi lain")
    print("yang mungkin sedang menggunakan kamera.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Kamera berhasil dibuka!")

# ==========================================
# 3. TAMPILKAN KAMERA
# ==========================================

window_name = "Exercise 4.14 - Image Classification"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# ==========================================
# 4. MUAT MODEL
# ==========================================

model = None
preprocess_input = None
decode_predictions = None

try:
    print(f"\nMemuat model {name}...")
    print("Jika baru pertama kali, bobot model mungkin perlu diunduh.")

    from classification_models.tfkeras import Classifiers
    from tensorflow.keras.applications.imagenet_utils import (
        decode_predictions as decode
    )

    decode_predictions = decode

    clf, preprocess_input = Classifiers.get(name)

    model = clf(
        input_shape=(sz, sz, 3),
        weights="imagenet",
        classes=1000
    )

    print(f"Model {name} berhasil dimuat!")

except Exception as e:
    print("\nGAGAL MEMUAT MODEL:")
    print(repr(e))
    print("Kamera tetap dapat digunakan tanpa prediksi.")

# ==========================================
# 5. PREDIKSI KAMERA
# ==========================================

last_prediction_time = 0
prediction_interval = 1.0

prediction_text = "Menunggu prediksi"
confidence = 0.0

try:
    while True:

        ret, cam_frame = camera.read()

        if not ret:
            print("Gagal membaca gambar dari kamera.")
            break

        current_time = time.time()

        # Prediksi setiap satu detik
        if (
            model is not None
            and current_time - last_prediction_time >= prediction_interval
        ):
            last_prediction_time = current_time

            try:
                # BGR menjadi RGB
                frame = cv2.cvtColor(
                    cam_frame, cv2.COLOR_BGR2RGB
                )

                # Resize sesuai kebutuhan model
                frame = cv2.resize(frame, (sz, sz))

                # Siapkan array gambar
                image = np.asarray(
                    frame, dtype=np.float32
                )
                image = np.expand_dims(image, axis=0)

                # Preprocessing
                image = preprocess_input(image)

                # Prediksi
                preds = model.predict(image, verbose=0)

                label = decode_predictions(
                    preds, top=1
                )[0][0]

                prediction_text = label[1]
                confidence = float(label[2])

                print(
                    f"Prediksi: {prediction_text} "
                    f"| Confidence: {confidence:.2%}"
                )

            except Exception as e:
                prediction_text = "Prediksi gagal"
                print("Error prediksi:", repr(e))

        # ==================================
        # 6. TAMPILKAN INFORMASI
        # ==================================

        cv2.rectangle(
            cam_frame,
            (0, 0),
            (cam_frame.shape[1], 85),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            cam_frame,
            f"Model: {name}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            cam_frame,
            f"Object: {prediction_text}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            cam_frame,
            f"Confidence: {confidence:.1%}",
            (10, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1
        )

        cv2.imshow(window_name, cam_frame)

        # ESC atau Q untuk keluar
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Kamera berhasil ditutup.")
