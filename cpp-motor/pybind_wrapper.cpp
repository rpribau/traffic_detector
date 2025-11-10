#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // Para convertir std::map a dict
#include <pybind11/numpy.h>    // Para convertir cv::Mat a numpy
#include "src/VehicleCounter.h"

namespace py = pybind11;

/**
 * @brief Convierte un cv::Mat (BGR, 3 canales) a un numpy array.
 * * Esta función crea una *copia* de los datos, asegurando que Python
 * sea el propietario de la memoria.
 */
py::array mat_to_nparray(const cv::Mat& mat) {
    if (mat.empty()) {
        // CORRECCIÓN 1: Usar un std::vector explícito para la forma (shape)
        std::vector<size_t> empty_shape = {0, 0, 3}; // Shape (0,0,3)
        return py::array_t<unsigned char>(empty_shape);
    }

    // CORRECCIÓN 2: Crear un std::vector explícito para la forma (shape)
    std::vector<size_t> shape = {
        static_cast<size_t>(mat.rows),
        static_cast<size_t>(mat.cols),
        static_cast<size_t>(mat.channels())
    };

    // Crear un nuevo array de numpy (py::array) del tamaño y tipo correctos
    // Esta llamada de constructor (solo con la forma) sí es válida y crea un 
    // array contiguo.
    py::array_t<unsigned char> np_array = py::array_t<unsigned char>(shape);

    // Copiar los datos de cv::Mat al buffer del array de numpy
    // Asumimos que la Mat de entrada (clonada) es contigua
    memcpy(np_array.mutable_data(), mat.data, mat.total() * mat.elemSize());

    return np_array;
}

PYBIND11_MODULE(motor_contador, m) {
    m.doc() = "Módulo C++ para conteo de vehículos usando TensorRT";

    py::class_<VehicleCounter>(m, "VehicleCounter")
        .def(py::init<const std::string&, const std::string&>(), 
             py::arg("onnx_model_path"), py::arg("trt_model_path") = "")
        
        .def("start_processing", &VehicleCounter::start_processing, 
             py::arg("video_path"), 
             "Inicia el procesamiento de video en un hilo separado.")
        
        .def("stop_processing", &VehicleCounter::stop_processing,
             "Detiene el hilo de procesamiento.")
        
        .def("get_latest_frame", 
             [](VehicleCounter& self) {
                 cv::Mat frame = self.get_latest_frame();
                 if (frame.empty()) {
                     // Devuelve un array numpy vacío si no hay frame
                     return mat_to_nparray(frame);
                 }
                 return mat_to_nparray(frame);
             }, 
             py::return_value_policy::move, // Mover el array a Python
             "Obtiene el último frame procesado como un array numpy (BGR).")
        
        .def("get_counts", &VehicleCounter::get_counts,
             py::return_value_policy::move, // Mover el dict a Python
             "Obtiene un diccionario con los conteos actuales.");
}
