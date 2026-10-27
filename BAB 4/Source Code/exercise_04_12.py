
# Exercise 4.12 - Webcam Image Classification with DenseNet Added

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import cv2
import threading
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import densenet

# Ukuran gambar
image_size = 224

print("TensorFlow:", tf.__version__, flush=True)

# Variabel global
latest_frame = None
prediction_label = "Loading..."
prediction_score = 0.0
predict_time = 0.0
status_text = "Memuat model..."

frame_lock = threading.Lock()
prediction_lock = threading.Lock()
stop_event = threading.Event()


# Membuka kamera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Kamera tidak dapat dibuka!", flush=True)
    raise SystemExit

camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Kamera berhasil dibuka!", flush=True)


# Thread untuk membaca kamera
def camera_worker():
    global latest_frame

    while not stop_event.is_set():
        ok, cam_frame = camera.read()

        if ok:
            with frame_lock:
                latest_frame = cam_frame.copy()
        else:
            time.sleep(0.05)


# Thread untuk klasifikasi DenseNet121
def prediction_worker():
    global prediction_label
    global prediction_score
    global predict_time
    global status_text

    try:
        print("1. Memuat model DenseNet121...", flush=True)

        model = densenet.DenseNet121(weights="imagenet")

        print("2. Model DenseNet121 berhasil dimuat!", flush=True)

        # Menguji model terlebih dahulu
        test_image = np.zeros(
            (1, image_size, image_size, 3),
            dtype=np.float32
        )

        test_image = densenet.preprocess_input(test_image)

        print("3. Menguji prediksi model...", flush=True)

        test_result = model.predict(
            test_image,
            batch_size=1,
            verbose=0
        )

        print("4. Tes prediksi berhasil!", flush=True)
        print("Bentuk output:", test_result.shape, flush=True)

        if test_result.shape != (1, 1000):
            raise ValueError(
                "Output model tidak sesuai: "
                + str(test_result.shape)
            )

        with prediction_lock:
            status_text = "Model siap"

        print("5. Memulai klasifikasi webcam...", flush=True)

        while not stop_event.is_set():

            # Mengambil frame terbaru
            with frame_lock:
                if latest_frame is None:
                    cam_frame = None
                else:
                    cam_frame = latest_frame.copy()

            if cam_frame is None:
                time.sleep(0.1)
                continue

            # Mengubah ukuran gambar
            frame = cv2.resize(
                cam_frame,
                (image_size, image_size)
            )

            # Mengubah gambar menjadi array
            numpy_image = img_to_array(frame)

            image_batch = np.expand_dims(
                numpy_image,
                axis=0
            )

            # Preprocessing khusus DenseNet121
            processed_image = densenet.preprocess_input(
                image_batch.copy()
            )

            # Melakukan prediksi
            start = time.time()

            predictions = model.predict(
                processed_image,
                batch_size=1,
                verbose=0
            )

            elapsed_time = time.time() - start

            # Mengambil tiga prediksi teratas
            results = decode_predictions(
                predictions,
                top=3
            )

            best_result = results[0][0]

            new_label = best_result[1]
            new_score = float(best_result[2]) * 100

            # Memperbarui hasil klasifikasi
            with prediction_lock:
                prediction_label = new_label
                prediction_score = new_score
                predict_time = elapsed_time
                status_text = "Klasifikasi berhasil"

            print("\nHasil klasifikasi:", flush=True)

            for result in results[0]:
                print(
                    "{}: {:.2f}%".format(
                        result[1],
                        float(result[2]) * 100
                    ),
                    flush=True
                )

            print(
                "Waktu prediksi: {:.3f} detik".format(
                    elapsed_time
                ),
                flush=True
            )

            # Jeda dua detik sebelum prediksi berikutnya
            for _ in range(20):
                if stop_event.is_set():
                    break
                time.sleep(0.1)

    except Exception as e:
        with prediction_lock:
            status_text = "Error pada model"

        print(
            "ERROR DenseNet121:",
            repr(e),
            flush=True
        )


# Menjalankan thread kamera
camera_thread = threading.Thread(
    target=camera_worker,
    daemon=True
)

# Menjalankan thread prediksi
prediction_thread = threading.Thread(
    target=prediction_worker,
    daemon=True
)

camera_thread.start()
prediction_thread.start()

print("Tekan ESC untuk keluar.", flush=True)


# Menampilkan kamera dan hasil klasifikasi
try:
    while True:

        # Mengambil frame terbaru
        with frame_lock:
            if latest_frame is None:
                cam_frame = None
            else:
                cam_frame = latest_frame.copy()

        if cam_frame is None:
            cv2.waitKey(1)
            time.sleep(0.01)
            continue

        # Mengambil hasil prediksi
        with prediction_lock:
            label_text = prediction_label
            score_text = prediction_score
            elapsed_time = predict_time
            current_status = status_text

        # Menampilkan nama objek dan probabilitas
        cv2.putText(
            cam_frame,
            "DenseNet: {} {:.1f}%".format(
                label_text,
                score_text
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2
        )

        # Menampilkan waktu prediksi
        cv2.putText(
            cam_frame,
            "Time: {:.3f} s".format(
                elapsed_time
            ),
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # Menampilkan status model
        cv2.putText(
            cam_frame,
            current_status,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.imshow("DenseNet121 Webcam Classification", cam_frame)

        # ESC untuk keluar
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop_event.set()

    camera.release()
    cv2.destroyAllWindows()

    print("Kamera ditutup.", flush=True)
