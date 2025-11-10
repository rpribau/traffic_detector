#pragma once

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <opencv2/opencv.hpp>
#include "yolov8.h" // Incluimos el YoloV8 que nos diste
#include "tracking/CentroidTracker.h"

class VehicleCounter {
public:
    VehicleCounter(const std::string& onnx_model_path, const std::string& trt_model_path);
    ~VehicleCounter();

    // --- Controles del Hilo ---
    // Inicia el bucle de procesamiento de video en un hilo separado
    void start_processing(const std::string& video_path);
    // Detiene el bucle de procesamiento
    void stop_processing();

    // --- Interfaz con Python (Thread-Safe) ---
    // Obtiene el último frame procesado (con dibujos)
    cv::Mat get_latest_frame();
    // Obtiene el mapa de conteos (ej. {"car": 10, "bus": 2})
    std::map<std::string, int> get_counts();

private:
    // El bucle principal que corre en m_processing_thread
    void processing_loop();

    // --- Componentes ---
    std::unique_ptr<YoloV8> m_yolo;
    YoloV8Config m_config;
    std::unique_ptr<CentroidTracker> m_tracker;
    cv::VideoCapture m_cap;

    // --- Estado y Lógica de Conteo ---
    std::map<std::string, int> m_counts;
    std::map<int, cv::Point> m_track_history; // Historial de centroides (ID -> Point)
    int m_line_y; // La posición Y de la línea de conteo

    // --- Datos compartidos y Mutex ---
    cv::Mat m_latest_frame;
    std::mutex m_frame_mutex;
    std::mutex m_counts_mutex;

    // --- Control del Hilo ---
    std::thread m_processing_thread;
    std::atomic<bool> m_is_running;
};
