#include "VehicleCounter.h"
#include <iostream>

VehicleCounter::VehicleCounter(const std::string& onnx_model_path, const std::string& trt_model_path) 
    : m_is_running(false), m_line_y(400) // Configura la línea de conteo en Y=400 (ajustar luego)
{
    // 1. Configurar YOLOv8
    // Usamos la configuración por defecto de YoloV8Config
    m_config.precision = Precision::FP16; 
    
    // Filtramos solo las clases que nos interesan
    // NOTA: Esto asume que el modelo ONNX fue entrenado en COCO.
    // Si tu modelo tiene clases diferentes, debes cambiarlas aquí.
    m_config.classNames = {"person", "bicycle", "car", "motorcycle", "bus", "truck"};

    // 2. Inicializar YoloV8 y Tracker
    // Usamos el constructor de YoloV8
    m_yolo = std::make_unique<YoloV8>(onnx_model_path, trt_model_path, m_config);
    m_tracker = std::make_unique<CentroidTracker>(100.0); // Max 100px de distancia

    std::cout << "Motor C++ VehicleCounter inicializado." << std::endl;
}

VehicleCounter::~VehicleCounter() {
    stop_processing();
}

void VehicleCounter::start_processing(const std::string& video_path) {
    if (m_is_running) {
        std::cout << "Advertencia: El procesamiento ya está en curso." << std::endl;
        return;
    }

    m_cap.open(video_path);
    if (!m_cap.isOpened()) {
        std::cerr << "Error: No se pudo abrir el video " << video_path << std::endl;
        return;
    }

    m_is_running = true;
    // Iniciar el hilo de procesamiento
    m_processing_thread = std::thread(&VehicleCounter::processing_loop, this);
    std::cout << "Procesamiento iniciado en hilo separado." << std::endl;
}

void VehicleCounter::stop_processing() {
    m_is_running = false;
    if (m_processing_thread.joinable()) {
        m_processing_thread.join();
        std::cout << "Hilo de procesamiento detenido." << std::endl;
    }
    if (m_cap.isOpened()) {
        m_cap.release();
    }
}

void VehicleCounter::processing_loop() {
    cv::Mat frame;
    while (m_is_running && m_cap.read(frame)) {
        if (frame.empty()) {
            std::cout << "Frame vacío, terminando bucle." << std::endl;
            break;
        }

        // 1. Detección de Objetos
        // Usamos detectObjects de YoloV8
        auto detections = m_yolo->detectObjects(frame);

        // 2. Tracking de Objetos
        auto tracked_objects = m_tracker->update(detections);

        // 3. Lógica de Conteo y Dibujo
        std::vector<Object> objects_to_draw;
        cv::Scalar line_color(0, 0, 255); // Línea roja

        for (const auto& tobj : tracked_objects) {
            std::string label = m_config.classNames[tobj.detection.label];
            
            // Lógica de conteo (cruce de línea)
            if (m_track_history.count(tobj.id)) {
                cv::Point prev_centroid = m_track_history[tobj.id];
                if (prev_centroid.y < m_line_y && tobj.centroid.y >= m_line_y) {
                    std::lock_guard<std::mutex> lock(m_counts_mutex);
                    m_counts[label]++;
                    std::cout << "CONTEO: " << label << " ID: " << tobj.id << std::endl;
                }
            }
            m_track_history[tobj.id] = tobj.centroid;
            
            // Dibujar ID del tracker
            std::string id_text = std::to_string(tobj.id);
            cv::putText(frame, id_text, 
                        cv::Point(tobj.detection.rect.x, tobj.detection.rect.y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            
            objects_to_draw.push_back(tobj.detection);
        }

        // Dibujar cajas (usando la función del repo)
        //
        m_yolo->drawObjectLabels(frame, objects_to_draw);

        // Dibujar la línea de conteo
        cv::line(frame, cv::Point(0, m_line_y), cv::Point(frame.cols, m_line_y), line_color, 2);

        // 4. Guardar frame para la UI (Thread-Safe)
        {
            std::lock_guard<std::mutex> lock(m_frame_mutex);
            m_latest_frame = frame.clone();
        }
    }

    m_is_running = false;
    std::cout << "Bucle de procesamiento finalizado." << std::endl;
}

cv::Mat VehicleCounter::get_latest_frame() {
    std::lock_guard<std::mutex> lock(m_frame_mutex);
    cv::Mat frame_copy = m_latest_frame.clone();
    m_latest_frame.release(); // Limpiamos para no enviar el mismo frame dos veces
    return frame_copy;
}

std::map<std::string, int> VehicleCounter::get_counts() {
    std::lock_guard<std::mutex> lock(m_counts_mutex);
    return m_counts; // Devuelve una copia
}