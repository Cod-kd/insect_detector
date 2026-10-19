# Insect Detector

Edge AI alapú kártevőfelismerő rendszer. Kameraképből felismeri az **egeret, patkányt, molylepkét és csótányt**, a megerősített észleléseket pedig egy valós idejű, telefonról is elérhető webes naplóban jeleníti meg.

Az inferencia **TensorFlow Lite Micro** motorral fut, így a detektor később mikrokontrolleres (alacsony fogyasztású) hardverre is átültethető. Jelenleg Docker konténerben, PC-n fut.

## Architektúra

```
┌──────────────────┐  JPEG (POST /frame)  ┌─────────────────────┐  JSON (POST /log)  ┌──────────────────┐
│ camera_sender.py │ ───────────────────▶ │ detector (Docker)   │ ─────────────────▶ │ server.py        │
│ webcam / fájlok  │       :8080          │ TFLite Micro + C++  │       :5000        │ Flask + dashboard│
└──────────────────┘                      └─────────────────────┘                    └──────────────────┘
                                                                                       ▲  böngésző / telefon
                                                                                       │  (SSE élő frissítés)
```

| Komponens | Fájl | Feladat |
|---|---|---|
| Képküldő | `camera_sender.py` | Webkamera vagy képfájl/mappa képkockáit küldi a detektornak, állítható FPS-sel |
| Detektor | `detector_micro.cpp` | Kép dekódolása, inferencia, téves riasztás-szűrés, log küldése |
| Log szerver | `server.py` | Fogadja és tárolja az észleléseket, élő HTML dashboardot szolgál ki |

## Funkciók

- 6 osztályos képosztályozó modell (`models/labels.txt`): `egér`, `patkány`, `molylepke`, `csótány`, `ember`, `háttér`
- Int8 / uint8 / float32 kvantált modellek támogatása
- **Téves riasztás-szűrés:**
  - konfidencia-küszöb (alapértelmezés: 90%)
  - N-of-M megerősítés: M egymást követő képkockából legalább N-nek ugyanazt kell mutatnia
- **Csendes címkék:** az embert és a hátteret felismeri, de nem naplózza (adatvédelem)
- Élő, mobilbarát dashboard: időbélyeg, címke, konfidencia, kivágott képrészlet, szűrés, szüneteltetés
- Futásidejű konfiguráció a `detector.env` fájlon keresztül, újrafordítás nélkül

## Könyvtárszerkezet

```
.
├── camera_sender.py          # képküldő kliens (host gépen fut)
├── detector_micro.cpp        # TFLite Micro detektor (Dockerben fut)
├── detector.env              # detektor konfiguráció
├── server.py                 # Flask log szerver + dashboard
├── models/
│   ├── model.tflite          # betanított, kvantált modell (~790 KB, 224×224 bemenet)
│   ├── labels.txt            # osztályok sorrendje
│   ├── check.py              # a modell által használt műveletek listázása
│   └── INSTRUCTION.md        # model.h generálása
└── docker-build/
    ├── Dockerfile            # multi-stage build (TFLM statikus lib + detektor)
    ├── patch_compatibility.py
    ├── _build.sh             # image építése
    └── _run.sh               # konténer indítása
```

## Követelmények

- Docker
- `git`, `python3`, `make` (a `create_tflm_tree.py` igényli)
- `xxd` (opcionális: ha nincs telepítve, a `_build.sh` Pythonnal generálja a `model.h`-t)
- Python csomagok a host gépen:
  ```bash
  pip install flask opencv-python requests
  ```

## Telepítés és futtatás

### 1. Függőségek és modell előkészítése

A projekt könyvtárában:

```bash
# TFLite Micro letöltése és standalone fa generálása
git clone --depth=1 https://github.com/tensorflow/tflite-micro.git
cd tflite-micro && python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py ../tflm_standalone
cd .. && rm -rf tflite-micro

# A modell beégetése C headerbe (a detektor ezt fordítja bele a binárisba)
xxd -i models/model.tflite > models/model.h
```

> A `tflm_standalone/` és a `models/model.h` a `.gitignore` része, ezért ezeket minden friss klón után újra elő kell állítani. A `_build.sh` a `model.h`-t automatikusan legenerálja, ha hiányzik vagy régebbi a `model.tflite`-nál. A `tflm_standalone/` hiánya esetén hibaüzenettel leáll.

### 2. Log szerver indítása

```bash
python server.py
```

A dashboard a `http://<PC_IP>:5000` címen érhető el, ugyanarról a WiFi-hálózatról telefonról is.

### 3. Detektor építése és indítása (Docker)

```bash
./docker-build/_build.sh && ./docker-build/_run.sh
```

> Mindkét szkript a `kamikou/insect-detector` image-nevet használja. Helyi build nélkül a Docker Hubról is letölthető: `docker pull kamikou/insect-detector`, utána elég a `_run.sh`.

### 4. Képek küldése

```bash
python camera_sender.py                         # alapértelmezett webkamera
python camera_sender.py --camera 1              # másik kamera
python camera_sender.py --file kep.jpg          # egyetlen kép
python camera_sender.py --file mappa/ --loop    # mappa képei folyamatosan
python camera_sender.py --fps 2 --url http://localhost:8080/frame
```

> A webkamerás mód a `CAP_DSHOW` backendet használja, ami Windows-specifikus.

## Konfiguráció (`detector.env`)

| Változó | Alapérték | Leírás |
|---|---|---|
| `INPUT_W`, `INPUT_H` | `224` | A modell bemeneti mérete |
| `TENSOR_ARENA_KB` | `1400` | TFLM memória-aréna; ha az `AllocateTensors` hibát ad, növeld |
| `FLASK_URL` | `http://host.docker.internal:5000/log` | A log szerver végpontja |
| `LISTEN_PORT` | `8080` | A detektor képfogadó portja |
| `CONFIDENCE_THRESH` | `0.90` | Minimális konfidencia egy találathoz |
| `CONFIRM_N`, `CONFIRM_M` | `3`, `5` | N-of-M megerősítés |
| `LOG_LABELS` | `Egér,Patkány,Molylepke,Csótány` | Ezekről készül log |
| `SILENT_LABELS` | `Ember,Háttér` | Felismeri, de nem naplózza |

> **Fontos:** a `LOG_LABELS` és a `SILENT_LABELS` együtt, ebben a sorrendben adja ki a modell osztálysorrendjét (lásd `models/labels.txt`). Ha a sorrend eltér, a címkék összekeverednek.

## API

**Detektor (`:8080`)**

| Metódus | Útvonal | Leírás |
|---|---|---|
| `POST` | `/frame` | JPEG képkocka (`Content-Type: image/jpeg`) |

**Log szerver (`:5000`)**

| Metódus | Útvonal | Leírás |
|---|---|---|
| `POST` | `/log` | Észlelés fogadása (JSON: `label`, `confidence`, `position`, `image_b64`) |
| `GET` | `/logs` | Eddigi naplóbejegyzések, a legújabb elöl |
| `GET` | `/stream` | Server-Sent Events élő stream |
| `GET` | `/` | HTML dashboard |

## Modell

- Formátum: TFLite, kvantált, kb. 790 KB
- Bemenet: `[1, 224, 224, 3]` RGB
- Kimenet: 6 osztály (lásd `models/labels.txt`)
- Használt műveletek: Quantize, Dequantize, Pad, Conv2D, DepthwiseConv2D, Add, Mean, FullyConnected, Softmax. Új modell esetén ellenőrizd a `models/check.py` szkripttel, és ha szükséges, bővítsd a `register_ops()` függvényt.

## Ismert korlátok

- **Osztályozás, nem objektumdetekció:** a modell az egész képre ad egy címkét. A naplóban szereplő pozíció mindig a kép közepe, a mentett kép pedig fix középső kivágás.
- A naplók csak memóriában tárolódnak (legfeljebb 500 bejegyzés), a szerver újraindításakor elvesznek.
- A dashboardon és a végpontokon nincs hitelesítés, ezért csak megbízható helyi hálózaton használd.
- A tanítási folyamat és az adathalmaz nem része a repónak, és mért pontossági adat sincs.
- Éjszakai (IR) működés még nincs.

## Tervek

- [ ] Valódi objektumdetekció (például FOMO vagy kis YOLO-variáns), több objektum és pozíció kezelése
- [ ] Tartós tárolás (SQLite) és CSV/PDF riportexport (HACCP-dokumentációhoz)
- [ ] Portolás célhardverre (ESP32-S3 / Raspberry Pi) IR kamerával
- [ ] Tanító pipeline és kiértékelési eredmények a repóban
- [ ] Hitelesítés a dashboardon
