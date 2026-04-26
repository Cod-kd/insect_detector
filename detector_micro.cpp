/*
 * detector_micro.cpp  —  TFLite Micro + HTTP képfogadás + Flask log
 *
 * A képforrást a camera_sender.py adja HTTP POST-on keresztül.
 * camera_sender.py használata:
 *   python camera_sender.py            ← webcam (alapértelmezett)
 *   python camera_sender.py --file kep.jpg     ← egyetlen képfájl
 *   python camera_sender.py --file kep.jpg --loop  ← folyamatosan ismételve
 */

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "models/model.h"
#define MODEL_DATA models_model_tflite
#define MODEL_LEN  models_model_tflite_len

#include <opencv2/opencv.hpp>
#include <curl/curl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <stdint.h>
#include <vector>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <pthread.h>

/* ═══════════════════════════════════════════════════════════════
   KONFIGURÁCIÓ
   ═══════════════════════════════════════════════════════════════ */
#define FLASK_URL         "http://host.docker.internal:5000/log"
#define LISTEN_PORT       8080
#define INPUT_W           224
#define INPUT_H           224
#define CONFIDENCE_THRESH 0.70f

#define TENSOR_ARENA_KB   1400
static uint8_t tensor_arena[TENSOR_ARENA_KB * 1024];

static const char *LABELS[] = {
    "Patkány",
    "Egér",
    "Csótány",
    "Hangya",
    "Molylepke",
    "Hangya raj"
};
static const int NUM_LABELS = 6;

/* ═══════════════════════════════════════════════════════════════
   GLOBÁLIS ÁLLAPOT  —  HTTP szerver ↔ inferencia szál
   ═══════════════════════════════════════════════════════════════ */
static pthread_mutex_t frame_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  frame_cond  = PTHREAD_COND_INITIALIZER;
static std::vector<uint8_t> pending_frame;
static bool frame_ready = false;

/* ═══════════════════════════════════════════════════════════════
   BASE64
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
   HTTP POST  →  Flask
   ═══════════════════════════════════════════════════════════════ */
static size_t curl_sink(void*, size_t s, size_t n, void*) { return s*n; }

static void post_log(const char *label, float confidence,
                     int cx, int cy, const cv::Mat &crop)
{
    std::vector<uint8_t> jpeg_buf;
    cv::imencode(".jpg", crop, jpeg_buf, {cv::IMWRITE_JPEG_QUALITY, 80});
    std::string img_b64 = base64_encode(jpeg_buf.data(), jpeg_buf.size());

    std::time_t t = std::time(nullptr);
    std::tm tm_info{};
    localtime_r(&t, &tm_info);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tm_info);

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
   HTTP SZERVER  —  fogadja a JPEG képkockákat POST /frame-en
   ═══════════════════════════════════════════════════════════════ */
static void *http_server_thread(void *) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(LISTEN_PORT);
    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 5);

    printf("[http] Képfogadó szerver fut: port %d\n", LISTEN_PORT);

    while (true) {
        int client = accept(server_fd, nullptr, nullptr);
        if (client < 0) continue;

        std::vector<uint8_t> buf(1024 * 1024);  /* 1MB buffer */
        ssize_t total = 0;

        while (total < (ssize_t)buf.size()) {
            ssize_t n = recv(client, buf.data() + total,
                             buf.size() - total, 0);
            if (n <= 0) break;
            total += n;

            std::string hdr(buf.begin(), buf.begin() + std::min(total, (ssize_t)2048));
            size_t hdr_end = hdr.find("\r\n\r\n");
            if (hdr_end == std::string::npos) continue;

            size_t cl_pos = hdr.find("Content-Length: ");
            if (cl_pos == std::string::npos) break;
            int content_len = std::stoi(hdr.substr(cl_pos + 16));
            size_t body_start = hdr_end + 4;
            size_t need = body_start + content_len;

            buf.resize(need + 1);
            while (total < (ssize_t)need) {
                n = recv(client, buf.data() + total, need - total, 0);
                if (n <= 0) break;
                total += n;
            }

            std::vector<uint8_t> jpeg(buf.begin() + body_start,
                                      buf.begin() + body_start + content_len);
            pthread_mutex_lock(&frame_mutex);
            pending_frame = std::move(jpeg);
            frame_ready   = true;
            pthread_cond_signal(&frame_cond);
            pthread_mutex_unlock(&frame_mutex);

            const char *resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
            send(client, resp, strlen(resp), 0);
            break;
        }
        close(client);
    }
    return nullptr;
}

/* ═══════════════════════════════════════════════════════════════
   INPUT TENSOR
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
                *dst++ = px[0]; *dst++ = px[1]; *dst++ = px[2];
            }
    } else if (tensor->type == kTfLiteInt8) {
        int8_t *dst = tensor->data.int8;
        float scale = tensor->params.scale;
        int   zp    = tensor->params.zero_point;
        for (int y = 0; y < INPUT_H; y++)
            for (int x = 0; x < INPUT_W; x++) {
                cv::Vec3b px = rgb.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; c++) {
                    int q = (int)roundf(px[c] / (255.0f * scale)) + zp;
                    q = q > 127 ? 127 : (q < -128 ? -128 : q);
                    *dst++ = (int8_t)q;
                }
            }
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
    } else if (tensor->type == kTfLiteInt8) {
        const int8_t *out = tensor->data.int8;
        float scale = tensor->params.scale;
        int   zp    = tensor->params.zero_point;
        float vals[NUM_LABELS], mx = -1e9f;
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
    } else if (tensor->type == kTfLiteUInt8) {
        const uint8_t *out = tensor->data.uint8;
        float scale = tensor->params.scale;
        int   zp    = tensor->params.zero_point;
        uint8_t braw = 0;
        for (int i = 0; i < n; i++)
            if (out[i] > braw) { braw = out[i]; *best_idx = i; }
        *best_conf = (braw - zp) * scale;
    }
}

/* ═══════════════════════════════════════════════════════════════
   OP RESOLVER
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
        fprintf(stderr, "[hiba] Schema verzió eltérés\n");
        return 1;
    }

    tflite::MicroMutableOpResolver<9> resolver;
    register_ops(resolver);
    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena));

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "[hiba] AllocateTensors sikertelen — növeld TENSOR_ARENA_KB!\n");
        return 1;
    }
    printf("[init] Arena felhasznált: %zu KB\n",
           interpreter.arena_used_bytes() / 1024);

    TfLiteTensor       *input  = interpreter.input(0);
    const TfLiteTensor *output = interpreter.output(0);
    printf("[init] Input  [1,%d,%d,3] típus=%d\n", INPUT_H, INPUT_W, input->type);
    printf("[init] Output [1,%d] típus=%d\n", NUM_LABELS, output->type);

    curl_global_init(CURL_GLOBAL_DEFAULT);

    pthread_t srv_thread;
    pthread_create(&srv_thread, nullptr, http_server_thread, nullptr);

    printf("[ready] Várakozás képkockákra (port %d)...\n", LISTEN_PORT);
    printf("        camera_sender.py --camera   ← webcam\n");
    printf("        camera_sender.py --file kep.jpg  ← képfájl\n\n");

    while (true) {
        pthread_mutex_lock(&frame_mutex);
        while (!frame_ready)
            pthread_cond_wait(&frame_cond, &frame_mutex);
        std::vector<uint8_t> jpeg = std::move(pending_frame);
        frame_ready = false;
        pthread_mutex_unlock(&frame_mutex);

        cv::Mat frame = cv::imdecode(jpeg, cv::IMREAD_COLOR);
        if (frame.empty()) continue;

        fill_input_tensor(input, frame);

        if (interpreter.Invoke() == kTfLiteOk) {
            int   best_idx;
            float best_conf;
            read_output(output, &best_idx, &best_conf);

            const char *label = (best_idx < NUM_LABELS)
                ? LABELS[best_idx] : "ismeretlen";

            printf("[detekció] %-15s  %.1f%%\n", label, best_conf * 100.0f);

            if (best_conf >= CONFIDENCE_THRESH) {
                int cx = frame.cols / 2, cy = frame.rows / 2;
                int cw = frame.cols / 2, ch = frame.rows / 2;
                cv::Rect roi(cx - cw/2, cy - ch/2, cw, ch);
                cv::Mat crop = frame(roi).clone();
                post_log(label, best_conf, cx, cy, crop);
                printf("[log küldve] %s @ (%d,%d)\n", label, cx, cy);
            }
        }
    }

    curl_global_cleanup();
    return 0;
}
