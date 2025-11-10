import sys
import numpy as np
import cv2 

from PySide6.QtCore import (
    Qt, 
    QThread, 
    Slot, 
    QSize, 
    QMetaObject, 
    Q_ARG
)
from PySide6.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QPushButton, 
    QComboBox, 
    QLineEdit, 
    QFileDialog, 
    QLabel, 
    QTabWidget, 
    QFormLayout, 
    QSpacerItem, 
    QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QIcon

# Importamos nuestro worker de hilo separado
from .video_processor_worker import VideoProcessorWorker
# --- (NUEVO) Importamos la nueva pestaña de Métricas ---
from .metrics_tab import MetricsTab

class MainWindow(QMainWindow):

    def __init__(self, motor_module):
        super().__init__()
        self.motor_module = motor_module
        
        self.setWindowTitle("Contador de Vehículos (C++ TensorRT + Python UI)")
        self.setGeometry(100, 100, 1280, 720)

        # --- Variables de estado ---
        self.video_path = ""
        self.model_path = "models/yolov8n.onnx" 

        # --- Hilo de procesamiento ---
        self.worker_thread = None
        self.worker = None

        # --- (NUEVO) Etiquetas para el Sidebar ---
        self.sb_lbl_car = QLabel("0")
        self.sb_lbl_truck = QLabel("0")
        self.sb_lbl_bus = QLabel("0")
        self.sb_lbl_person = QLabel("0")

        # --- Construir la UI ---
        self.init_ui()
        self.setup_worker_thread()

    def init_ui(self):
        """Crea todos los widgets de la UI."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 1. Sidebar de Controles ---
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar")
        sidebar_widget.setFixedWidth(280)
        
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        
        # --- Controles del Sidebar ---
        sidebar_layout.addWidget(QLabel("CONFIGURACIÓN"))
        
        self.btn_open_video = QPushButton("Abrir Video")
        self.btn_open_video.setObjectName("secondary")
        self.btn_open_video.setIcon(QIcon.fromTheme("document-open"))
        
        self.le_video_path = QLineEdit("No se ha seleccionado video")
        self.le_video_path.setReadOnly(True)

        sidebar_layout.addWidget(self.btn_open_video)
        sidebar_layout.addWidget(self.le_video_path)
        
        sidebar_layout.addWidget(QLabel("UBICACIÓN DE CÁMARA"))
        self.combo_camera = QComboBox()
        self.camera_locations = {
            "Camara Av. Revolucion 1": (25.6601, -100.2821),
            "Av. Revolucion 2": (25.6555, -100.2790),
            "Av. Luis Elizondo": (25.6515, -100.2905)
        }
        self.combo_camera.addItems(self.camera_locations.keys())
        sidebar_layout.addWidget(self.combo_camera)

        sidebar_layout.addWidget(QLabel("ACCIONES"))
        self.btn_start = QPushButton("Iniciar Procesamiento")
        self.btn_start.setIcon(QIcon.fromTheme("media-playback-start"))
        
        self.btn_export = QPushButton("Generar Reporte")
        self.btn_export.setIcon(QIcon.fromTheme("document-save"))
        self.btn_export.setObjectName("secondary")
        
        sidebar_layout.addWidget(self.btn_start)
        sidebar_layout.addWidget(self.btn_export)

        # --- (NUEVO) Métricas en Tiempo Real en Sidebar ---
        sidebar_layout.addWidget(QLabel("CONTEO EN VIVO"))
        
        metrics_form_layout = QFormLayout()
        
        # Conectar las etiquetas del sidebar (creadas en __init__)
        self.sb_lbl_car.setObjectName("sidebar_metrics_value")
        self.sb_lbl_truck.setObjectName("sidebar_metrics_value")
        self.sb_lbl_bus.setObjectName("sidebar_metrics_value")
        self.sb_lbl_person.setObjectName("sidebar_metrics_value")

        car_label = QLabel("Carros:")
        car_label.setObjectName("sidebar_metrics_label")
        metrics_form_layout.addRow(car_label, self.sb_lbl_car)

        truck_label = QLabel("Camiones:")
        truck_label.setObjectName("sidebar_metrics_label")
        metrics_form_layout.addRow(truck_label, self.sb_lbl_truck)

        bus_label = QLabel("Autobuses:")
        bus_label.setObjectName("sidebar_metrics_label")
        metrics_form_layout.addRow(bus_label, self.sb_lbl_bus)

        person_label = QLabel("Peatones:")
        person_label.setObjectName("sidebar_metrics_label")
        metrics_form_layout.addRow(person_label, self.sb_lbl_person)
        
        sidebar_layout.addLayout(metrics_form_layout)

        # Espaciador
        sidebar_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        main_layout.addWidget(sidebar_widget)

        # --- 2. Área de Contenido (Pestañas) ---
        content_widget = QWidget()
        content_widget.setObjectName("content_area")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        self.tabs = QTabWidget()
        content_layout.addWidget(self.tabs)
        
        # --- Pestaña 1: Video en Vivo ---
        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel("El video se mostrará aquí")
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)
        self.tabs.addTab(video_tab, "Video en Tiempo Real")

        # --- Pestaña 2: Métricas (AHORA RECONSTRUIDA) ---
        self.metrics_tab_widget = MetricsTab(self.camera_locations)
        self.tabs.addTab(self.metrics_tab_widget, "Métricas y Reportes")

        main_layout.addWidget(content_widget, 1)

        # --- 3. Barra de Estado ---
        self.statusBar().showMessage("Listo.")

        # --- Conectar botones ---
        self.btn_open_video.clicked.connect(self.open_video_file)
        self.btn_start.clicked.connect(self.start_processing)
        # Conectar el combobox de la cámara al mapa
        self.combo_camera.currentTextChanged.connect(self.metrics_tab_widget.center_map_on)


    def setup_worker_thread(self):
        self.worker_thread = QThread()
        self.worker = VideoProcessorWorker(self.motor_module)
        self.worker.moveToThread(self.worker_thread)
        self.worker.new_frame_ready.connect(self.update_video_frame)
        self.worker.new_counts_ready.connect(self.update_counts)
        self.worker.status_updated.connect(lambda msg: self.statusBar().showMessage(msg))
        self.worker.finished.connect(self.processing_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    # -----------------------------------------------------------------
    # --- SLOTS Y FUNCIONES DE MANEJO DE EVENTOS ---
    # -----------------------------------------------------------------

    @Slot()
    def open_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Video", "", "Archivos de Video (*.mp4 *.avi *.mov)"
        )
        if path:
            self.video_path = path
            self.le_video_path.setText(path.split('/')[-1])
            self.statusBar().showMessage(f"Video seleccionado: {path}")

    @Slot()
    def start_processing(self):
        if self.video_path == "":
            self.statusBar().showMessage("Error: Por favor, selecciona un video primero.")
            return
            
        self.btn_start.setEnabled(False)
        self.btn_open_video.setEnabled(False)
        self.statusBar().showMessage("Iniciando procesamiento...")

        QMetaObject.invokeMethod(
            self.worker,
            "start_processing",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, self.video_path),
            Q_ARG(str, self.model_path)
        )

    @Slot()
    def processing_finished(self):
        self.statusBar().showMessage("Procesamiento finalizado.")
        self.btn_start.setEnabled(True)
        self.btn_open_video.setEnabled(True)
        
    @Slot(np.ndarray)
    def update_video_frame(self, frame_array):
        if frame_array.size == 0:
            return
            
        h, w, ch = frame_array.shape
        bytes_per_line = ch * w
        
        if ch == 3:
            frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            q_image = QImage(frame_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
        )

    @Slot(dict)
    def update_counts(self, counts):
        """
        (ACTUALIZADO)
        Actualiza AMBAS etiquetas: las del sidebar y las de la pestaña principal.
        """
        car_count = counts.get("car", 0)
        truck_count = counts.get("truck", 0)
        bus_count = counts.get("bus", 0)
        person_count = counts.get("person", 0)
        
        # --- 1. Actualizar Sidebar ---
        self.sb_lbl_car.setText(str(car_count))
        self.sb_lbl_truck.setText(str(truck_count))
        self.sb_lbl_bus.setText(str(bus_count))
        self.sb_lbl_person.setText(str(person_count))
        
        # --- 2. Actualizar Pestaña de Métricas (Gráficas) ---
        # (El QFormLayout de la pestaña principal se eliminó,
        #  así que ya no necesitamos actualizarlo.
        #  En el futuro, aquí es donde llamaríamos a 
        #  self.metrics_tab_widget.update_graphs(counts) )

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker.stop_processing()
            self.worker_thread.quit()
            self.worker_thread.wait(5000)
        event.accept()