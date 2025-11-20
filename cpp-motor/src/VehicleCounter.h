#pragma once

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "yolov8.h" 
#include "tracking/CentroidTracker.h"

// Estructura mejorada para líneas y checkpoints
struct CountingLine {
    cv::Point p1;
    cv::Point p2;
    std::string label;      // Clave para el conteo (ej: "Revolucion_NS")
    bool is_checkpoint;     // True = Cuenta vehículos, False = Solo dibujo
    cv::Scalar color;       // Color de la línea
};

class VehicleCounter {
public:
    VehicleCounter(const std::string& onnx_model_path, const std::string& trt_model_path);
    ~VehicleCounter();

    void start_processing(const std::string& video_path);
    void stop_processing();

    cv::Mat get_latest_frame();
    std::map<std::string, int> get_counts();

private:
    void processing_loop();

    // Función auxiliar matemática para intersección de segmentos
    bool segments_intersect(cv::Point a, cv::Point b, cv::Point c, cv::Point d);
    int ccw(cv::Point a, cv::Point b, cv::Point c);

    std::unique_ptr<YoloV8> m_yolo;
    YoloV8Config m_config;
    std::unique_ptr<CentroidTracker> m_tracker;
    cv::VideoCapture m_cap;

    std::map<std::string, int> m_counts;
    
    // Historial simple del último punto (necesario para calcular intersección)
    std::map<int, cv::Point> m_prev_centroids; 

    std::map<int, std::vector<cv::Point>> m_track_paths;
    std::vector<CountingLine> m_counting_lines;
    // Registro de qué IDs ya fueron contados en qué línea para no contarlos doble
    std::map<int, std::vector<std::string>> m_counted_ids; 

    cv::Mat m_latest_frame;
    std::mutex m_frame_mutex;
    std::mutex m_counts_mutex;

    std::thread m_processing_thread;
    std::atomic<bool> m_is_running;
};