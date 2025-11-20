import sys
import numpy as np
import cv2 

from PySide6.QtCore import Qt, QThread, Slot, QSize, QMetaObject, Q_ARG
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QComboBox, QLineEdit, QFileDialog, QLabel, QTabWidget, 
    QFormLayout, QSpacerItem, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QIcon

from .video_processor_worker import VideoProcessorWorker
from .metrics_tab import MetricsTab

class MainWindow(QMainWindow):

    def __init__(self, motor_module):
        super().__init__()
        self.motor_module = motor_module
        self.setWindowTitle("Monitor de Tráfico Monterrey - YOLOv8")
        self.setGeometry(100, 100, 1280, 720)

        self.video_path = ""
        self.model_path = "models/yolov8n.onnx" 
        self.worker_thread = None
        self.worker = None

        # --- NUEVAS ETIQUETAS PARA CARRILES ---
        self.lbl_cov_oe = QLabel("0") # Covarrubias Oeste-Este
        self.lbl_rev_ns = QLabel("0") # Revolucion Norte-Sur
        self.lbl_cov_eo = QLabel("0") # Covarrubias Este-Oeste

        self.init_ui()
        self.setup_worker_thread()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Sidebar ---
        sidebar_widget = QWidget()
        sidebar_widget.setObjectName("sidebar")
        sidebar_widget.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(15, 15, 15, 15)
        
        sidebar_layout.addWidget(QLabel("CONFIGURACIÓN"))
        self.btn_open_video = QPushButton("Abrir Video")
        self.btn_open_video.setObjectName("secondary")
        self.le_video_path = QLineEdit("Sin video seleccionado")
        self.le_video_path.setReadOnly(True)
        sidebar_layout.addWidget(self.btn_open_video)
        sidebar_layout.addWidget(self.le_video_path)
        
        sidebar_layout.addWidget(QLabel("CÁMARA"))
        self.combo_camera = QComboBox()
        self.camera_locations = {
             "Av. Revolucion / Av. R. Covarrubias": {
                "coords": (25.652859, -100.277200),
                "direction": "Intersección Principal"
            }
        }
        self.combo_camera.addItems(self.camera_locations.keys())
        sidebar_layout.addWidget(self.combo_camera)

        sidebar_layout.addWidget(QLabel("CONTROLES"))
        self.btn_start = QPushButton("Iniciar")
        self.btn_export = QPushButton("Reporte")
        self.btn_export.setObjectName("secondary")
        sidebar_layout.addWidget(self.btn_start)
        sidebar_layout.addWidget(self.btn_export)

        # --- MÉTRICAS DE CARRILES ---
        sidebar_layout.addWidget(QLabel("FLUJO VEHICULAR"))
        metrics_form = QFormLayout()
        
        # Estilo para los números
        for lbl in [self.lbl_cov_oe, self.lbl_rev_ns, self.lbl_cov_eo]:
            lbl.setObjectName("sidebar_metrics_value")

        # Etiquetas descriptivas
        l1 = QLabel("Covarrubias (Oeste -> Este):")
        l1.setObjectName("sidebar_metrics_label")
        l1.setWordWrap(True)
        metrics_form.addRow(l1, self.lbl_cov_oe)

        l2 = QLabel("Revolución (Norte -> Sur):")
        l2.setObjectName("sidebar_metrics_label")
        l2.setWordWrap(True)
        metrics_form.addRow(l2, self.lbl_rev_ns)

        l3 = QLabel("Covarrubias (Este -> Oeste):")
        l3.setObjectName("sidebar_metrics_label")
        l3.setWordWrap(True)
        metrics_form.addRow(l3, self.lbl_cov_eo)
        
        sidebar_layout.addLayout(metrics_form)
        sidebar_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        main_layout.addWidget(sidebar_widget)

        # --- Contenido Principal ---
        content_widget = QWidget()
        content_widget.setObjectName("content_area")
        content_layout = QVBoxLayout(content_widget)
        self.tabs = QTabWidget()
        
        # Tab Video
        video_tab = QWidget()
        v_layout = QVBoxLayout(video_tab)
        v_layout.setContentsMargins(0,0,0,0)
        self.video_label = QLabel("Vista previa")
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        v_layout.addWidget(self.video_label)
        self.tabs.addTab(video_tab, "Monitor en Vivo")

        # Tab Mapa
        self.metrics_tab = MetricsTab(self.camera_locations)
        self.tabs.addTab(self.metrics_tab, "Mapa y Datos")
        
        content_layout.addWidget(self.tabs)
        main_layout.addWidget(content_widget, 1)

        # Conexiones
        self.btn_open_video.clicked.connect(self.open_video_file)
        self.btn_start.clicked.connect(self.start_processing)

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

    @Slot()
    def open_video_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir", "", "Video (*.mp4 *.avi)")
        if path:
            self.video_path = path
            self.le_video_path.setText(path.split('/')[-1])

    @Slot()
    def start_processing(self):
        if not self.video_path: return
        self.btn_start.setEnabled(False)
        self.btn_open_video.setEnabled(False)
        QMetaObject.invokeMethod(self.worker, "start_processing", 
                               Qt.ConnectionType.QueuedConnection, 
                               Q_ARG(str, self.video_path), Q_ARG(str, self.model_path))

    @Slot()
    def processing_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_open_video.setEnabled(True)

    @Slot(np.ndarray)
    def update_video_frame(self, frame):
        if frame.size == 0: return
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    @Slot(dict)
    def update_counts(self, counts):
        # Actualizamos los contadores usando las claves EXACTAS definidas en C++
        self.lbl_cov_oe.setText(str(counts.get("Covarrubias (Oeste-Este)", 0)))
        self.lbl_rev_ns.setText(str(counts.get("Revolucion (Norte-Sur)", 0)))
        self.lbl_cov_eo.setText(str(counts.get("Covarrubias (Este-Oeste)", 0)))

    def closeEvent(self, event):
        if self.worker_thread.isRunning():
            self.worker.stop_processing()
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()