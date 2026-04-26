/*
 * detector_micro.cpp  —  TFLite Micro + PC kamera + Flask HTTP log
 *
 * FORDÍTÁS (Docker-ben, lásd Dockerfile):
 *   g++ detector_micro.cpp tflm_srcs/*.cc \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -Itflm_includes \
 *       -Ithird_party/flatbuffers/include \
 *       -lcurl -lm -lpthread \
 *       -DTF_LITE_STATIC_MEMORY \
 *       -std=c++17 -O2 -o detector
 *
 * FÁJLSTRUKTÚRA:
 *   project/
 *   ├── detector_micro.cpp
 *   ├── Dockerfile
 *   ├── setup_tflm.sh
 *   ├── models/
 *       ├── model.tflite
 *       └── model.h          ← xxd -i models/model.tflite > models/model.h
 */

/* ── TFLite Micro fejlécek ──────────────────────────────────── */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ── Generált modell tömb (xxd -i models/model.tflite > models/model.h) */
#include "models/model.h"
/*
 * Az xxd a fájl útvonalából generálja a nevet, pl.:
 *   models/model.tflite  →  models_model_tflite[] + models_model_tflite_len
 * Ha eltér, módosítsd az alábbi két sort:
 */
#define MODEL_DATA models_model_tflite
#define MODEL_LEN  models_model_tflite_len

/* ── OpenCV ─────────────────────────────────────────────────── */
#include <opencv2/opencv.hpp>

/* ── libcurl ────────────────────────────────────────────────── */
#include <curl/curl.h>

/* ── Standard ───────────────────────────────────────────────── */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <stdint.h>
#include <vector>
#include <string>

/* ═══════════════════════════════════════════════════════════════
   KONFIGURÁCIÓ
   ═══════════════════════════════════════════════════════════════ */
#define FLASK_URL         "http://host.docker.internal:5000/log"
/*                         ↑ Docker-ből a host PC Flask szervere
 *                           Linux Docker esetén: http://172.17.0.1:5000/log
 *                           Ha ugyanazon gépen fut: http://127.0.0.1:5000/log
 */
#define CAMERA_INDEX      0
#define INPUT_W           224
#define INPUT_H           224
#define CONFIDENCE_THRESH 0.70f    /* 70% alatti találatot nem loggolunk */
#define INFERENCE_EVERY_N 10       /* minden 10. képkockán fut a modell  */

#define TENSOR_ARENA_KB   1400
static uint8_t tensor_arena[TENSOR_ARENA_KB * 1024];

/* Label lista — a modell osztályainak sorrendjében */
static const char *LABELS[] = {
    "Patkány",
    "Egér",
    "Csótány",
    "Hangya",
    "Molylepke",
    "Hangya raj"
};
static const int NUM_LABELS = 6;   /* ← 6 label van! */

/* ═══════════════════════════════════════════════════════════════
   BASE64 KÓDOLÁS  (libb64 nélkül, beépített)
   ═══════════════════════════════════════════════════════════════ */
static const char B64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const uint8_t *src, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t b  = (uint32_t)src[i] << 16;
        if (i+1 < len) b |= (uint32_t)src[i+1] << 8;
        if (i+2 < len) b |= (uint32_t)src[i+2];
        out += B64[(b >> 18) & 0x3F];
        out += B64[(b >> 12) & 0x3F];
        out += (i+1 < len) ? B64[(b >>  6) & 0x3F] : '=';
        out += (i+2 < len) ? B64[(b      ) & 0x3F] : '=';
    }
    return out;
}

/* ═══════════════════════════════════════════════════════════════
   HTTP POST  →  Flask  (multipart: JSON meta + JPEG kép)
   ═══════════════════════════════════════════════════════════════ */
static size_t curl_sink(void*, size_t s, size_t n, void*) { return s*n; }

/*
 * Küld a Flask /log végpontjára:
 *   {
 *     "timestamp": "2024-01-15 14:23:01",
 *     "label":     "Egér",
 *     "confidence": 0.9231,
 *     "position":  {"x": 210, "y": 180},
 *     "image_b64": "<JPEG base64>"
 *   }
 *
 * A "position" a képkocka közepe — osztályozó modellnél nincs
 * valódi bounding box, ezért a teljes frame közepét küldjük.
 * Ha a jövőben detection modellre váltasz, itt add meg a bbox közepét.
 */
static void post_log(const char *label, float confidence,
                     int cx, int cy,
                     const cv::Mat &crop)
{
    /* JPEG kódolás memóriába */
    std::vector<uint8_t> jpeg_buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
    cv::imencode(".jpg", crop, jpeg_buf, params);
    std::string img_b64 = base64_encode(jpeg_buf.data(), jpeg_buf.size());

    /* Timestamp */
    std::time_t t = std::time(nullptr);

    std::tm tm_info{};

    #ifdef _WIN32
        localtime_s(&tm_info, &t);
    #else
        localtime_r(&t, &tm_info);
    #endif

    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_info);

    /* JSON összerakása — a base64 string lehet nagy, std::string kell */
    std::string json = std::string("{")
        + "\"timestamp\":\"" + ts + "\","
        + "\"label\":\""     + label + "\","
        + "\"confidence\":"  + std::to_string(confidence) + ","
        + "\"position\":{\"x\":" + std::to_string(cx)
                      + ",\"y\":" + std::to_string(cy) + "},"
        + "\"image_b64\":\"" + img_b64 + "\""
        + "}";

    CURL *curl = curl_easy_init();
    if (!curl) return;

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL,            FLASK_URL);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,     json.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,  (long)json.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  curl_sink);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        3L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 2L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK)
        fprintf(stderr, "[curl] %s\n", curl_easy_strerror(res));

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}

/* ═══════════════════════════════════════════════════════════════
   KÉP  →  INPUT TENSOR
   ═══════════════════════════════════════════════════════════════ */
static void fill_input_tensor(TfLiteTensor *tensor, const cv::Mat &frame) {
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(INPUT_W, INPUT_H));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    if (tensor->type == kTfLiteFloat32) {
        float *dst = tensor->data.f;
        for (int y = 0; y < INPUT_H; y++)
            for (int x = 0; x < INPUT_W; x++) {
                cv::Vec3b px = rgb.at<cv::Vec3b>(y, x);
                *dst++ = px[0] / 255.0f;
                *dst++ = px[1] / 255.0f;
                *dst++ = px[2] / 255.0f;
            }
    } else if (tensor->type == kTfLiteUInt8) {
        uint8_t *dst = tensor->data.uint8;
        for (int y = 0; y < INPUT_H; y++)
            for (int x = 0; x < INPUT_W; x++) {
                cv::Vec3b px = rgb.at<cv::Vec3b>(y, x);
                *dst++ = px[0];
                *dst++ = px[1];
                *dst++ = px[2];
            }
    } else if (tensor->type == kTfLiteInt8) {
        /* INT8 kvantált bemenet — a legtöbb quantized MobileNet ilyen */
        int8_t *dst = tensor->data.int8;
        float scale = tensor->params.scale;
        int   zp    = tensor->params.zero_point;
        for (int y = 0; y < INPUT_H; y++)
            for (int x = 0; x < INPUT_W; x++) {
                cv::Vec3b px = rgb.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; c++) {
                    int q = (int)roundf(px[c] / (255.0f * scale)) + zp;
                    if (q > 127)  q = 127;
                    if (q < -128) q = -128;
                    *dst++ = (int8_t)q;
                }
            }
    } else {
        fprintf(stderr, "[warn] Ismeretlen input tensor típus: %d\n", tensor->type);
    }
}

/* ═══════════════════════════════════════════════════════════════
   KIMENET OLVASÁSA
   ═══════════════════════════════════════════════════════════════ */
static void read_output(const TfLiteTensor *tensor,
                        int *best_idx, float *best_conf)
{
    int n = tensor->dims->data[tensor->dims->size - 1];
    if (n > NUM_LABELS) n = NUM_LABELS;
    *best_idx = 0; *best_conf = 0.0f;

    if (tensor->type == kTfLiteFloat32) {
        const float *out = tensor->data.f;
        float mx = out[0];
        for (int i = 1; i < n; i++) if (out[i] > mx) mx = out[i];
        float sum = 0;
        for (int i = 0; i < n; i++) sum += expf(out[i] - mx);
        float best = -1e9f;
        for (int i = 0; i < n; i++) {
            float p = expf(out[i] - mx) / sum;
            if (p > best) { best = p; *best_idx = i; }
        }
        *best_conf = best;

    } else if (tensor->type == kTfLiteUInt8) {
        const uint8_t *out = tensor->data.uint8;
        float scale = tensor->params.scale;
        int   zp    = tensor->params.zero_point;
        uint8_t braw = 0;
        for (int i = 0; i < n; i++)
            if (out[i] > braw) { braw = out[i]; *best_idx = i; }
        *best_conf = (braw - zp) * scale;

    } else if (tensor->type == kTfLiteInt8) {
        /* INT8 kimenet — a DEQUANTIZE op előtti állapot */
        const int8_t *out = tensor->data.int8;
        float scale = tensor->params.scale;
        int   zp    = tensor->params.zero_point;
        /* Softmax az int8 értékekből */
        float vals[NUM_LABELS];
        float mx = -1e9f;
        for (int i = 0; i < n; i++) {
            vals[i] = (out[i] - zp) * scale;
            if (vals[i] > mx) mx = vals[i];
        }
        float sum = 0;
        for (int i = 0; i < n; i++) sum += expf(vals[i] - mx);
        float best = -1e9f;
        for (int i = 0; i < n; i++) {
            float p = expf(vals[i] - mx) / sum;
            if (p > best) { best = p; *best_idx = i; }
        }
        *best_conf = best;
    }
}

/* ═══════════════════════════════════════════════════════════════
   OP RESOLVER  —  a modell tényleges op listája alapján
   ═══════════════════════════════════════════════════════════════ */
static void register_ops(tflite::MicroMutableOpResolver<9> &resolver) {
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddPad();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
}

/* ═══════════════════════════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════════════════════════ */
int main(void) {
    tflite::InitializeTarget();

    printf("[init] Modell betöltése (%u bájt)...\n", MODEL_LEN);
    const tflite::Model *model = tflite::GetModel(MODEL_DATA);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        fprintf(stderr, "[hiba] Schema verzió eltérés: modell=%d runtime=%d\n",
                model->version(), TFLITE_SCHEMA_VERSION);
        return 1;
    }

    tflite::MicroMutableOpResolver<9> resolver;
    register_ops(resolver);

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "[hiba] AllocateTensors sikertelen — "
                        "növeld a TENSOR_ARENA_KB értékét!\n");
        return 1;
    }
    printf("[init] Arena felhasznált: %zu KB\n",
           interpreter.arena_used_bytes() / 1024);

    TfLiteTensor       *input  = interpreter.input(0);
    const TfLiteTensor *output = interpreter.output(0);

    printf("[init] Input  shape: [%d,%d,%d,%d] típus=%d\n",
           input->dims->data[0], input->dims->data[1],
           input->dims->data[2], input->dims->data[3], input->type);
    printf("[init] Output shape: [%d,%d] típus=%d\n",
           output->dims->data[0], output->dims->data[1], output->type);

    /* Kamera */
    cv::VideoCapture cap(CAMERA_INDEX);
    if (!cap.isOpened()) {
        fprintf(stderr, "[hiba] Kamera nem érhető el (index=%d)\n", CAMERA_INDEX);
        return 1;
    }
    int cam_w = 640, cam_h = 480;
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cam_w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_h);
    printf("[kamera] Megnyitva (%dx%d). ESC = kilépés.\n\n", cam_w, cam_h);

    curl_global_init(CURL_GLOBAL_DEFAULT);

    cv::Mat frame;
    int frame_n = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;
        frame_n++;

        if (frame_n % INFERENCE_EVERY_N == 0) {
            fill_input_tensor(input, frame);

            if (interpreter.Invoke() == kTfLiteOk) {
                int   best_idx;
                float best_conf;
                read_output(output, &best_idx, &best_conf);

                const char *label = (best_idx < NUM_LABELS)
                    ? LABELS[best_idx] : "ismeretlen";

                printf("[detekció] %-15s  %.1f%%\n",
                       label, best_conf * 100.0f);

                /* Overlay a preview ablakra */
                char txt[80];
                snprintf(txt, sizeof(txt), "%s  %.0f%%",
                         label, best_conf * 100.0f);
                cv::putText(frame, txt, cv::Point(10, 36),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 230, 160), 2);

                if (best_conf >= CONFIDENCE_THRESH) {
                    /*
                     * Osztályozó modellnél nincs valódi bounding box.
                     * Pozícióként a frame közepét küldjük.
                     * Crop: a frame középső 50%-a (ahol valószínűleg az obj van)
                     */
                    int cx = frame.cols / 2;
                    int cy = frame.rows / 2;
                    int cw = frame.cols / 2;
                    int ch = frame.rows / 2;
                    cv::Rect roi(cx - cw/2, cy - ch/2, cw, ch);
                    cv::Mat crop = frame(roi).clone();

                    /* Piros keret a crop területén */
                    cv::rectangle(frame, roi, cv::Scalar(0, 0, 220), 2);

                    post_log(label, best_conf, cx, cy, crop);
                    printf("[log küldve] %s @ (%d,%d)\n", label, cx, cy);
                }
            }
        }

        cv::imshow("Detektor  [ESC = kilepes]", frame);
        if (cv::waitKey(1) == 27) break;
    }

    curl_global_cleanup();
    cap.release();
    cv::destroyAllWindows();
    printf("[vege] Leallitva.\n");
    return 0;
}
