#include "VehicleCounter.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <unordered_set>

// --- FUNCIONES MATEMÁTICAS AUXILIARES ---
int VehicleCounter::ccw(cv::Point a, cv::Point b, cv::Point c) {
    double val = (double)(b.x - a.x) * (c.y - a.y) - (double)(b.y - a.y) * (c.x - a.x);
    if (val > 0) return 1;
    if (val < 0) return -1;
    return 0;
}

bool VehicleCounter::segments_intersect(cv::Point a, cv::Point b, cv::Point c, cv::Point d) {
    return (ccw(a, b, c) * ccw(a, b, d) < 0) && (ccw(c, d, a) * ccw(c, d, b) < 0);
}

VehicleCounter::VehicleCounter(const std::string& onnx_model_path, const std::string& trt_model_path) 
    : m_is_running(false)
{
    m_config.precision = Precision::FP16; 
    
    m_yolo = std::make_unique<YoloV8>(onnx_model_path, trt_model_path, m_config);
    m_tracker = std::make_unique<CentroidTracker>(100.0); 

    // --- DEFINICIÓN DE LÍNEAS ---
    // 1. Carril Covarrubias (Oeste -> Este)
    m_counting_lines.push_back({{120, 100}, {230, 230}, "", false, cv::Scalar(0, 255, 0)});
    m_counting_lines.push_back({{170, 90}, {300, 170}, "", false, cv::Scalar(0, 255, 0)});
    m_counting_lines.push_back({{305, 140}, {230, 230}, "Covarrubias (Oeste-Este)", true, cv::Scalar(0, 0, 255)});

    // 2. Carril Revolucion (Norte -> Sur)
    m_counting_lines.push_back({{166, 90}, {40, 230}, "", false, cv::Scalar(255, 255, 0)});
    m_counting_lines.push_back({{220, 90}, {160, 230}, "", false, cv::Scalar(255, 255, 0)});
    m_counting_lines.push_back({{10, 140}, {185, 190}, "Revolucion (Norte-Sur)", true, cv::Scalar(0, 0, 255)});

    // 3. Carril Covarrubias (Este -> Oeste)
    m_counting_lines.push_back({{270, 110}, {170, 90}, "", false, cv::Scalar(148, 0, 211)});
    m_counting_lines.push_back({{270, 90}, {180, 80}, "", false, cv::Scalar(148, 0, 211)});
    m_counting_lines.push_back({{189, 65}, {169, 86}, "Covarrubias (Este-Oeste)", true, cv::Scalar(0, 0, 255)});

    std::cout << "Motor inicializado con cajas ajustadas." << std::endl;
}

VehicleCounter::~VehicleCounter() {
    stop_processing();
}

void VehicleCounter::start_processing(const std::string& video_path) {
    if (m_is_running) return;
    m_cap.open(video_path);
    if (!m_cap.isOpened()) {
        std::cerr << "Error al abrir video." << std::endl;
        return;
    }
    m_is_running = true;
    m_processing_thread = std::thread(&VehicleCounter::processing_loop, this);
}

void VehicleCounter::stop_processing() {
    m_is_running = false;
    if (m_processing_thread.joinable()) m_processing_thread.join();
    if (m_cap.isOpened()) m_cap.release();
}

void VehicleCounter::processing_loop() {
    cv::Mat frame;
    const int TARGET_FRAME_DURATION_MS = 67; 
    
    m_prev_centroids.clear();
    m_track_paths.clear();
    m_counted_ids.clear();

    static const std::unordered_set<std::string> allowed_classes = {
        "car", "truck", "bus", "motorcycle", "bicycle"
    };

    while (m_is_running && m_cap.read(frame)) {
        auto start_time = std::chrono::steady_clock::now();

        if (frame.empty()) break;

        cv::Size target_size(320, 240); 
        if (frame.size() != target_size) {
            cv::resize(frame, frame, target_size, 0, 0, cv::INTER_AREA);
        }

        // 1. Detección
        auto raw_detections = m_yolo->detectObjects(frame);

        // --- NUEVO: REDUCCIÓN DE CAJAS (SHRINK) ---
        // Reducimos el tamaño de la caja un 20% (0.2) para que se ajuste mejor al objeto
        // y el centroide sea más preciso, ignorando sombras periféricas.
        float shrink_amount = 0.2f; 

        for (auto& det : raw_detections) {
            float w = det.rect.width;
            float h = det.rect.height;
            
            // Nuevas dimensiones
            float new_w = w * (1.0f - shrink_amount);
            float new_h = h * (1.0f - shrink_amount);
            
            // Centrar la nueva caja dentro de la vieja
            float offset_x = (w - new_w) / 2.0f;
            float offset_y = (h - new_h) / 2.0f;

            det.rect.x += offset_x;
            det.rect.y += offset_y;
            det.rect.width = new_w;
            det.rect.height = new_h;
        }
        // -------------------------------------------

        std::vector<Object> filtered_detections;
        for (const auto& det : raw_detections) {
            if (det.label >= 0 && det.label < (int)m_config.classNames.size()) {
                if (allowed_classes.count(m_config.classNames[det.label])) {
                    filtered_detections.push_back(det);
                }
            }
        }

        // 2. Tracking
        auto tracked_objects = m_tracker->update(filtered_detections);

        std::vector<Object> objects_to_draw;
        
        // Limpieza de datos viejos
        for (auto it = m_track_paths.begin(); it != m_track_paths.end(); ) {
            bool found = false;
            for (const auto& tobj : tracked_objects) if (tobj.id == it->first) found = true;
            if (!found) {
                it = m_track_paths.erase(it);
                m_prev_centroids.erase(it->first);
            } else ++it;
        }

        for (const auto& tobj : tracked_objects) {
            m_track_paths[tobj.id].push_back(tobj.centroid);
            if (m_track_paths[tobj.id].size() > 20) m_track_paths[tobj.id].erase(m_track_paths[tobj.id].begin());

            // Lógica de Conteo
            if (m_prev_centroids.count(tobj.id)) {
                cv::Point prev = m_prev_centroids[tobj.id];
                cv::Point curr = tobj.centroid;

                for (const auto& line : m_counting_lines) {
                    if (line.is_checkpoint) {
                        if (segments_intersect(prev, curr, line.p1, line.p2)) {
                            
                            bool already_counted = false;
                            if (m_counted_ids.count(tobj.id)) {
                                for (const auto& counted_label : m_counted_ids[tobj.id]) {
                                    if (counted_label == line.label) {
                                        already_counted = true;
                                        break;
                                    }
                                }
                            }

                            if (!already_counted) {
                                std::lock_guard<std::mutex> lock(m_counts_mutex);
                                m_counts[line.label]++;
                                m_counted_ids[tobj.id].push_back(line.label);
                                std::cout << "CRUCE: " << line.label << " (ID: " << tobj.id << ")" << std::endl;
                                cv::circle(frame, curr, 10, cv::Scalar(255, 255, 255), -1);
                            }
                        }
                    }
                }
            }
            m_prev_centroids[tobj.id] = tobj.centroid;

            // Dibujar ID
            cv::putText(frame, std::to_string(tobj.id), tobj.centroid, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,0), 1);
            objects_to_draw.push_back(tobj.detection);
        }

        m_yolo->drawObjectLabels(frame, objects_to_draw, 1); 

        for (const auto& line : m_counting_lines) {
            cv::line(frame, line.p1, line.p2, line.color, line.is_checkpoint ? 3 : 2);
        }

        {
            std::lock_guard<std::mutex> lock(m_frame_mutex);
            m_latest_frame = frame.clone();
        }

        auto end_time = std::chrono::steady_clock::now();
        int sleep_ms = TARGET_FRAME_DURATION_MS - (int)std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    }
    m_is_running = false;
}

cv::Mat VehicleCounter::get_latest_frame() {
    std::lock_guard<std::mutex> lock(m_frame_mutex);
    if (m_latest_frame.empty()) return cv::Mat();
    return m_latest_frame.clone();
}

std::map<std::string, int> VehicleCounter::get_counts() {
    std::lock_guard<std::mutex> lock(m_counts_mutex);
    return m_counts;
}
