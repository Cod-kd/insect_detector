"""
camera_sender.py  —  Windows hoston fut, képet küld a Docker containernek

Használat:
  python camera_sender.py                        <- webcam (alapértelmezett)
  python camera_sender.py --camera 1             <- másik kamera index
  python camera_sender.py --file kep.jpg         <- egyetlen képfájl (egyszer)
  python camera_sender.py --file kep.jpg --loop  <- képfájl folyamatosan ismételve
  python camera_sender.py --file mappa/          <- mappa összes képe sorban

Telepítés:
  pip install opencv-python requests
"""

import cv2
import requests
import time
import argparse
import os
import sys

DETECTOR_URL = "http://localhost:8080/frame"
JPEG_QUALITY = 80

def parse_args():
    parser = argparse.ArgumentParser(description="Képküldő a Docker detektornak")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--camera", type=int, default=None,
                       metavar="INDEX",
                       help="Webcam index (alapértelmezett: 0)")
    group.add_argument("--file", type=str, default=None,
                       metavar="ÚTVONAL",
                       help="Képfájl vagy mappa útvonala")
    parser.add_argument("--loop", action="store_true",
                        help="Képfájl/mappa folyamatos ismétlése")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Küldési FPS (alapértelmezett: 5)")
    parser.add_argument("--url", type=str, default=DETECTOR_URL,
                        help=f"Detektor URL (alapértelmezett: {DETECTOR_URL})")
    return parser.parse_args()


def send_frame(session, url, frame, quality=JPEG_QUALITY):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False
    try:
        resp = session.post(
            url,
            data=buf.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            timeout=2
        )
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        print("[warn] Detektor nem érhető el — fut a container? (_run.sh)")
        return False
    except Exception as e:
        print(f"[warn] {e}")
        return False


def collect_images(path):
    """Mappa esetén összegyűjti a képfájlokat."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if not files:
            print(f"[hiba] Nem találtam képfájlt a mappában: {path}")
            sys.exit(1)
        return files
    else:
        print(f"[hiba] Nem létező útvonal: {path}")
        sys.exit(1)


def run_camera(args):
    idx = args.camera if args.camera is not None else 0
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[hiba] Kamera nem érhető el (index={idx})")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[kamera] Webcam #{idx} megnyitva → {args.url}")
    print(f"[kamera] {args.fps} FPS küldés | Q = kilépés\n")

    session  = requests.Session()
    interval = 1.0 / args.fps

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[warn] Üres képkocka")
            continue

        send_frame(session, args.url, frame)

        cv2.imshow("Küldő [Q = kilépés]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    cap.release()
    cv2.destroyAllWindows()


def run_file(args):
    files = collect_images(args.file)
    session  = requests.Session()
    interval = 1.0 / args.fps

    print(f"[fájl] {len(files)} kép → {args.url}")
    print(f"[fájl] {args.fps} FPS | loop={'igen' if args.loop else 'nem'} | Q = kilépés\n")

    while True:
        for path in files:
            frame = cv2.imread(path)
            if frame is None:
                print(f"[warn] Nem olvasható: {path}")
                continue

            t0 = time.time()
            ok = send_frame(session, args.url, frame)
            status = "✓" if ok else "✗"
            print(f"  {status} {os.path.basename(path)}")

            cv2.imshow("Küldő [Q = kilépés]", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

        if not args.loop:
            print("[kész] Minden kép elküldve.")
            break
        print("[loop] Újrakezdem...\n")

    cv2.destroyAllWindows()


def main():
    args = parse_args()

    if args.file is not None:
        run_file(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
