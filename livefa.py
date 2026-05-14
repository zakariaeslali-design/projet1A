# -*- coding: utf-8 -*-
"""Improved live face recognition script.

Features:
- Argument parsing for model/encoder/cascade/camera paths
- Robust path handling and logging
- Safer resource cleanup and error reporting
- Small UI improvements (colors, confidence format)
"""

from pathlib import Path
import argparse
import logging
import sys
import csv
from datetime import datetime
import time
from typing import Optional

import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
from keras_facenet import FaceNet
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live face recognition with attendance logging")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--model", type=str, default="svm_model_160x160.pkl", help="Path to trained SVM model pickle")
    p.add_argument("--encoder", type=str, default="label_encoder.pkl", help="Path to label encoder pickle")
    p.add_argument("--cascade", type=str, default=None, help="Path to Haarcascade XML (default: OpenCV bundled)")
    p.add_argument("--out", type=str, default="presences", help="Output directory for attendance CSV files")
    p.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold for recognition (0-1)")
    p.add_argument("--remark-delay", type=int, default=60, help="Seconds before re-marking same person")
    return p.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_pickle(path: Path, what: str):
    if not path.exists():
        logging.error("%s not found: %s", what, path)
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_cascade(cascade_arg: Optional[str]) -> cv.CascadeClassifier:
    if cascade_arg:
        cascade_path = Path(cascade_arg)
        if not cascade_path.exists():
            logging.warning("Provided cascade not found, falling back to OpenCV bundled cascade")
        else:
            return cv.CascadeClassifier(str(cascade_path))

    # Fallback to OpenCV data
    cascade_default = Path(cv.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_default.exists():
        raise FileNotFoundError("Haarcascade XML not found in OpenCV data")
    return cv.CascadeClassifier(str(cascade_default))


def ensure_csv(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    csv_file = out_dir / f"presence_{today}.csv"
    if not csv_file.exists():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Nom", "Date", "Heure", "Statut"])
    return csv_file


def main():
    args = parse_args()
    setup_logging()

    base = Path(__file__).resolve().parent
    model_path = (base / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    encoder_path = (base / args.encoder) if not Path(args.encoder).is_absolute() else Path(args.encoder)
    cascade_arg = args.cascade

    logging.info("Loading model and encoder...")
    try:
        model = load_pickle(model_path, "Model")
        encoder = load_pickle(encoder_path, "Label encoder")
    except Exception as e:
        logging.exception("Failed to load model or encoder: %s", e)
        sys.exit(1)

    logging.info("Classes: %s", getattr(encoder, "classes_", "<unknown>"))

    haarcascade = ensure_cascade(cascade_arg)

    csv_file = ensure_csv(Path(args.out))

    # Rate limiting dictionary
    already_marked: dict[str, datetime] = {}
    remark_delay = args.remark_delay

    def mark_presence(name: str) -> bool:
        now = datetime.now()
        last = already_marked.get(name)
        if last is not None and (now - last).total_seconds() < remark_delay:
            return False
        already_marked[name] = now
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Présent"])
        logging.info("Présence marquée: %s at %s", name, now.strftime("%H:%M:%S"))
        return True

    # Initialize FaceNet once (may take time)
    logging.info("Initializing FaceNet (this may take a moment)...")
    facenet = FaceNet()

    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened():
        logging.error("Cannot open camera index %s", args.camera)
        sys.exit(1)

    logging.info("Starting capture. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning("Empty frame, retrying...")
                time.sleep(0.1)
                continue

            frame = cv.flip(frame, 1)
            rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # safe crop with bounds
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = x + w, y + h
                face_rgb = rgb_img[y1:y2, x1:x2]
                try:
                    face_resized = cv.resize(face_rgb, (160, 160))
                except Exception:
                    continue

                emb = facenet.embeddings(np.expand_dims(face_resized, axis=0))

                probabilities = model.predict_proba(emb)[0]
                max_confidence = float(np.max(probabilities))
                face_pred = model.predict(emb)

                if max_confidence >= args.threshold:
                    final_name = encoder.inverse_transform(face_pred)[0]
                    color = (0, 255, 0)
                    just_marked = mark_presence(final_name)
                    if just_marked:
                        cv.putText(frame, "Présence marquée !", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    final_name = "Inconnu"
                    color = (0, 0, 255)

                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, f"{final_name} ({max_confidence:.1%})", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)

            # HUD: number of unique marked presences and last few names
            last_names = list(already_marked.keys())[-5:]
            hud_text = f"Présences: {len(already_marked)} | Derniers: {', '.join(last_names)}"
            cv.putText(frame, hud_text, (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv.imshow("Face Recognition - Presence", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
    finally:
        cap.release()
        cv.destroyAllWindows()
        logging.info("Session terminée. Présences sauvegardées dans: %s", csv_file)


if __name__ == "__main__":
    main()