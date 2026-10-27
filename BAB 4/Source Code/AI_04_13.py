
# Example 4.13 - VGG16 Real-Time Object Recognition

# Harus diatur sebelum import TensorFlow
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import threading
import time
import tensorflow as tf

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# =========================
# KONFIGURASI
# =========================
IMAGE_SIZE = 224
PREDICTION_INTERVAL = 1.0

latest_frame = None
latest_label = "Menunggu prediksi..."
frame_lock = threading.Lock()
stop_event = threading.Event()

# =========================
# PROSES PREDIKSI
# =========================
def prediction_worker(model):
    global latest_frame, latest_label

    last_prediction = 0

    while not stop_event.is_set():

        if time.time() - last_prediction < PREDICTION_INTERVAL:
            time.sleep(0.02)
            continue

        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        try:
            # OpenCV menggunakan BGR, model VGG16 memakai RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

            image = frame.astype(np.float32)
            image = np.expand_dims(image, axis=0)
            image = vgg16.preprocess_input(image)

            # Gunakan predict seperti tes yang berhasil
            predictions = model.predict(
                image,
                batch_size=1,
                verbose=0
            )

            result = decode_predictions(predictions, top=1)
            object_name = result[0][0][1]
            confidence = result[0][0][2] * 100

            label = f"{object_name}: {confidence:.1f}%"

            with frame_lock:
                latest_label = label

            print("Prediksi:", label, flush=True)

        except Exception as error:
            with frame_lock:
                latest_label = "Prediksi gagal"

            print("Kesalahan prediksi:", repr(error), flush=True)

        last_prediction = time.time()


# =========================
# PROGRAM UTAMA
# =========================
def main():
    global latest_frame, latest_label

    camera = None
    worker = None

    try:
        print("[1] Memuat model VGG16...", flush=True)

        model = vgg16.VGG16(weights="imagenet")

        print("[2] Model berhasil dimuat!", flush=True)

        # Warm-up model dengan prediksi yang sudah diuji
        print("[3] Menguji prediksi VGG16...", flush=True)

        dummy = np.zeros(
            (1, IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype=np.float32
        )
        dummy = vgg16.preprocess_input(dummy)

        model.predict(dummy, batch_size=1, verbose=0)

        print("[4] Prediksi siap!", flush=True)

        # Buka webcam
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():
            print("Kamera gagal dibuka!", flush=True)
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("[5] Kamera aktif!", flush=True)
        print("Tekan ESC untuk keluar.", flush=True)

        worker = threading.Thread(
            target=prediction_worker,
            args=(model,),
            daemon=True
        )
        worker.start()

        while True:
            ok, cam_frame = camera.read()

            if not ok:
                print("Gagal membaca frame kamera!", flush=True)
                break

            with frame_lock:
                latest_frame = cam_frame.copy()
                label = latest_label

            display_frame = cam_frame.copy()

            # Panel latar teks
            cv2.rectangle(
                display_frame,
                (0, 0),
                (display_frame.shape[1], 55),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                display_frame,
                label,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(
                "VGG16 - Real-Time Object Recognition",
                display_frame
            )

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\nProgram dihentikan.", flush=True)

    except Exception as error:
        print("Kesalahan program:", repr(error), flush=True)

    finally:
        stop_event.set()

        if worker is not None:
            worker.join(timeout=2)

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()
        print("Program selesai.", flush=True)


if __name__ == "__main__":
    main()
