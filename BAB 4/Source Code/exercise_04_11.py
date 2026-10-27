
# Exercise 4.11 - Webcam Image Classification with VGG19

import time
import cv2
import threading
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Ukuran gambar
image_size = 224

# Variabel global
latest_frame = None
prediction_label = "Loading..."
prediction_score = 0.0
predict_time = 0.0

frame_lock = threading.Lock()
prediction_lock = threading.Lock()
stop_event = threading.Event()


# Membuka kamera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Kamera tidak dapat dibuka!")
    raise SystemExit

print("Kamera berhasil dibuka!")


# Thread untuk membaca kamera
def camera_worker():
    global latest_frame

    while not stop_event.is_set():
        ret, cam_frame = camera.read()

        if ret:
            with frame_lock:
                latest_frame = cam_frame.copy()
        else:
            time.sleep(0.05)


# Thread untuk prediksi VGG19
def prediction_worker():
    global prediction_label, prediction_score, predict_time

    print("Memuat model VGG19...")

    try:
        model = vgg19.VGG19(weights="imagenet")
        print("Model VGG19 berhasil dimuat!")
        print("Mulai melakukan klasifikasi...")

    except Exception as e:
        print("Gagal memuat model VGG19:", e)
        stop_event.set()
        return

    while not stop_event.is_set():

        with frame_lock:
            if latest_frame is None:
                cam_frame = None
            else:
                cam_frame = latest_frame.copy()

        if cam_frame is None:
            time.sleep(0.1)
            continue

        try:
            # Resize gambar
            frame = cv2.resize(
                cam_frame,
                (image_size, image_size)
            )

            # Konversi gambar menjadi array
            numpy_image = img_to_array(frame)
            image_batch = np.expand_dims(
                numpy_image,
                axis=0
            )

            # Preprocessing VGG19
            processed_image = vgg19.preprocess_input(
                image_batch.copy()
            )

            # Prediksi
            start = time.time()

            predictions = model.predict(
                processed_image,
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

            # Memperbarui hasil
            with prediction_lock:
                prediction_label = new_label
                prediction_score = new_score
                predict_time = elapsed_time

            # Menampilkan hasil di terminal
            print("\nHasil klasifikasi:")

            for result in results[0]:
                print(
                    "{}: {:.2f}%".format(
                        result[1],
                        float(result[2]) * 100
                    )
                )

            print(
                "Waktu prediksi: {:.3f} detik".format(
                    elapsed_time
                )
            )

        except Exception as e:
            print("Error saat prediksi:", e)

        # Jeda dua detik sebelum prediksi berikutnya
        for _ in range(20):
            if stop_event.is_set():
                break
            time.sleep(0.1)


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

print("Klasifikasi dimulai. Tekan ESC untuk keluar.")


# Menampilkan kamera dan hasil klasifikasi
try:
    while True:

        with frame_lock:
            if latest_frame is None:
                cam_frame = None
            else:
                cam_frame = latest_frame.copy()

        if cam_frame is None:
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(0.01)
            continue

        # Mengambil hasil prediksi terbaru
        with prediction_lock:
            label_text = prediction_label
            score_text = prediction_score
            elapsed_time = predict_time

        # Menampilkan hasil VGG19
        cv2.putText(
            cam_frame,
            "VGG19: {}, {:.1f}%".format(
                label_text,
                score_text
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
            0.8,
            (255, 0, 0),
            2
        )

        cv2.imshow("video image", cam_frame)

        # ESC untuk keluar
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop_event.set()

    camera.release()
    cv2.destroyAllWindows()

    print("Kamera ditutup.")
