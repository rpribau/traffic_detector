cmake_minimum_required(VERSION 3.18)
project(motor_contador)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG")

# --- 1. Encontrar Dependencias ---

# Encontrar Pybind11
find_package(pybind11 REQUIRED)

# Encontrar OpenCV
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV no se encontró. Asegúrate de que esté instalado.")
endif()
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# --- 2. (NUEVO) Configurar CMAKE_MODULE_PATH ---
#
# ESTA ES LA CORRECCIÓN:
# Añadir los directorios 'cmake' de los subproyectos anidados ANTES
# de llamarlos. Esto permite que find_package(TensorRT) encuentre
# los scripts FindTensorRT.cmake.
#
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/YOLOv8-TensorRT-CPP/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/YOLOv8-TensorRT-CPP/libs/tensorrt-cpp-api/cmake")

# --- 3. Construir las Librerías Base ---

# Ahora que CMAKE_MODULE_PATH está configurado, esta llamada funcionará.
message(STATUS "Construyendo librerías base de YOLOv8 y TensorRT...")
add_subdirectory(libs/YOLOv8-TensorRT-CPP)

# --- 4. Construir Nuestro Módulo de Python ---

message(STATUS "Construyendo el módulo 'motor_contador' de Python...")

set(WRAPPER_SOURCES
    src/VehicleCounter.cpp
    pybind_wrapper.cpp
)

pybind11_add_module(motor_contador SHARED ${WRAPPER_SOURCES})

# --- 5. Enlazar Todo ---

target_include_directories(motor_contador PRIVATE
    ${pybind11_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    src/
    libs/YOLOv8-TensorRT-CPP/src
    libs/YOLOv8-TensorRT-CPP/libs/tensorrt-cpp-api/include
)

target_link_libraries(motor_contador PRIVATE
    YoloV8_TRT
    tensorrt_cpp_api
    ${OpenCV_LIBS}
)

target_link_libraries(motor_contador PRIVATE YoloV8_TRT)

message(STATUS "Configuración de CMake para 'motor_contador' completada.")
