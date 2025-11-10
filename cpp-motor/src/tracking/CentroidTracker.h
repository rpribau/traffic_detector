#pragma once

#include <vector>
#include <map>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "yolov8.h" // Para la estructura 'Object'

// Estructura simple para un objeto rastreado
struct TrackedObject {
    int id;
    Object detection; // La detección original de YOLO
    cv::Point centroid;
};

class CentroidTracker {
public:
    CentroidTracker(double max_distance = 75.0) : m_next_id(0), m_max_distance(max_distance) {}

    std::vector<TrackedObject> update(const std::vector<Object>& detections) {
        std::vector<TrackedObject> tracked_objects;

        // 1. Calcular centroides para las nuevas detecciones
        std::vector<cv::Point> new_centroids;
        for (const auto& det : detections) {
            int cx = det.rect.x + det.rect.width / 2;
            int cy = det.rect.y + det.rect.height / 2;
            new_centroids.push_back(cv::Point(cx, cy));
        }

        // Si no hay objetos rastreados, registrar todos los nuevos
        if (m_tracked_centroids.empty()) {
            for (size_t i = 0; i < detections.size(); ++i) {
                m_tracked_centroids[m_next_id] = new_centroids[i];
                tracked_objects.push_back({m_next_id, detections[i], new_centroids[i]});
                m_next_id++;
            }
            return tracked_objects;
        }

        // 2. Coincidencia simple (greedy) basada en distancia
        std::map<int, cv::Point> new_tracked_centroids;
        std::vector<bool> used_detections(detections.size(), false);

        for (auto const& [id, centroid] : m_tracked_centroids) {
            double min_dist = m_max_distance;
            int best_match_idx = -1;

            for (size_t i = 0; i < new_centroids.size(); ++i) {
                if (used_detections[i]) continue;
                
                double dist = cv::norm(centroid - new_centroids[i]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_match_idx = i;
                }
            }

            if (best_match_idx != -1) {
                // Objeto encontrado/coincidido
                new_tracked_centroids[id] = new_centroids[best_match_idx];
                tracked_objects.push_back({id, detections[best_match_idx], new_centroids[best_match_idx]});
                used_detections[best_match_idx] = true;
            }
            // Si no hay 'else', el objeto se "pierde" y no se agrega a new_tracked_centroids
        }

        // 3. Registrar nuevas detecciones que no coincidieron
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!used_detections[i]) {
                new_tracked_centroids[m_next_id] = new_centroids[i];
                tracked_objects.push_back({m_next_id, detections[i], new_centroids[i]});
                m_next_id++;
            }
        }

        m_tracked_centroids = new_tracked_centroids;
        return tracked_objects;
    }

private:
    int m_next_id;
    double m_max_distance;
    std::map<int, cv::Point> m_tracked_centroids; // ID -> Centroid
};
