import sys
from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QComboBox, QLineEdit, QFileDialog, 
    QLabel, QTabWidget, QFormLayout
)
from PySide6.QtGui import QPixmap, QImage
import numpy as np

from .video_processor_worker import VideoProcessorWorker

class MainWindow(QMainWindow):

    def __init__(self, motor_module):
        super().__init__()
        self.motor_module = motor_module
        
        self.setWindowTitle("Contador de Vehículos")
        self.setGeometry(100, 100, 1280, 720)

        # --- Variables de estado ---
        self.video_path = ""
        self.model_path = "models/yolov8n.onnx" # Ruta por defecto

        # --- Hilo de procesamiento ---
        self.worker_thread = None
        self.worker = None

        # --- Construir la UI ---
        self.init_ui()
        self.setup_worker_thread()

    def init_ui(self):
        """Crea todos los widgets de la UI."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- 1. Panel de Configuración ---
        config_widget = QWidget()
        config_layout = QHBoxLayout(config_widget)
        
        self.btn_open_video = QPushButton("1. Abrir Video")
        self.le_video_path = QLineEdit("No se ha seleccionado video")
        self.le_video_path.setReadOnly(True)
        
        self.combo_camera = QComboBox()
        self.combo_camera.addItems([
            "Camara Av. Revolucion 1", 
            "Av. Revolucion 2", 
            "Av. Luis Elizondo"
        ])
        
        self.btn_start = QPushButton("2. Iniciar Procesamiento")
        
        config_layout.addWidget(self.btn_open_video)
        config_layout.addWidget(self.le_video_path, 1) # El '1' le da más espacio
        config_layout.addWidget(QLabel("Ubicación:"))
        config_layout.addWidget(self.combo_camera)
        config_layout.addWidget(self.btn_start)
        
        main_layout.addWidget(config_widget)

        # --- 2. Pestañas de Vista ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1) # El '1' le da más espacio

        # --- Pestaña 1: Video en Vivo ---
        video_tab = QWidget()
        video_layout = QVBoxLayout(video_tab)
        self.video_label = QLabel("El video se mostrará aquí")
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_layout.addWidget(self.video_label)
        self.tabs.addTab(video_tab, "Video en Tiempo Real")

        # --- Pestaña 2: Métricas ---
        metrics_tab = QWidget()
        metrics_layout = QFormLayout(metrics_tab)
        self.lbl_count_car = QLabel("0")
        self.lbl_count_truck = QLabel("0")
        self.lbl_count_bus = QLabel("0")
        self.lbl_count_person = QLabel("0")
        self.lbl_count_total = QLabel("0")
        
        metrics_layout.addRow("Carros:", self.lbl_count_car)
        metrics_layout.addRow("Camiones de Carga:", self.lbl_count_truck)
        metrics_layout.addRow("Autobuses:", self.lbl_count_bus)
        metrics_layout.addRow("Peatones:", self.lbl_count_person)
        metrics_layout.addRow(QLabel("<strong>Total:</strong>"), self.lbl_count_total)
        
        self.btn_export = QPushButton("Generar Reporte (.parquet)")
        metrics_layout.addWidget(self.btn_export)
        self.tabs.addTab(metrics_tab, "Métricas y Reportes")

        # --- 3. Barra de Estado ---
        self.statusBar().showMessage("Listo.")

        # --- Conectar botones ---
        self.btn_open_video.clicked.connect(self.open_video_file)
        self.btn_start.clicked.connect(self.start_processing)
        # (self.btn_export.clicked.connect(self.export_report) -> Tarea)

    def setup_worker_thread(self):
        """Crea el Hilo y el Worker y conecta las señales."""
        self.worker_thread = QThread()
        self.worker = VideoProcessorWorker(self.motor_module)
        
        # Mueve el worker al hilo
        self.worker.moveToThread(self.worker_thread)

        # Conectar señales y slots
        # 1. Decirle al worker que empiece (desde UI)
        self.btn_start.clicked.connect(self.start_processing) 
        
        # 2. Recibir frames y conteos (desde el worker)
        self.worker.new_frame_ready.connect(self.update_video_frame)
        self.worker.new_counts_ready.connect(self.update_counts)
        
        # 3. Manejar estado y finalización
        self.worker.status_updated.connect(lambda msg: self.statusBar().showMessage(msg))
        self.worker.finished.connect(self.processing_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Iniciar el hilo (estará "dormido" esperando señales)
        self.worker_thread.start()

    @Slot()
    def open_video_file(self):
        """Abre un diálogo para seleccionar un archivo de video."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Video", "", "Archivos de Video (*.mp4 *.avi *.mov)"
        )
        if path:
            self.video_path = path
            self.le_video_path.setText(path)
            self.statusBar().showMessage(f"Video seleccionado: {path}")

    @Slot()
    def start_processing(self):
        if self.video_path == "":
            self.statusBar().showMessage("Error: Por favor, selecciona un video primero.")
            return
            
        self.btn_start.setEnabled(False)
        self.btn_open_video.setEnabled(False)
        self.statusBar().showMessage("Iniciando procesamiento...")

        # ¡Esto NO bloquea la UI! Emite una señal que el worker
        # recibirá en su propio hilo.
        # (Usamos una señal personalizada o QMetaObject.invokeMethod para ser
        #  seguros entre hilos, pero un connect directo a un slot
        #  de un worker en otro hilo es 'QueuedConnection' por defecto)
        
        # Para ser explícitos (y más robustos):
        QMetaObject.invokeMethod(
            self.worker,
            "start_processing",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, self.video_path),
            Q_ARG(str, self.model_path)
        )

    @Slot()
    def processing_finished(self):
        """Limpia la UI cuando el worker termina."""
        self.statusBar().showMessage("Procesamiento finalizado.")
        self.btn_start.setEnabled(True)
        self.btn_open_video.setEnabled(True)
        
    @Slot(np.ndarray)
    def update_video_frame(self, frame_array):
        """Convierte un array numpy (frame) en un QPixmap y lo muestra."""
        if frame_array.size == 0:
            return
            
        h, w, ch = frame_array.shape
        bytes_per_line = ch * w
        
        # Convertir BGR (OpenCV) a RGB (Qt)
        # Asumimos que el motor C++ nos devuelve BGR
        if ch == 3:
            frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            # Asumir que ya es un formato compatible si no es 3 canales
            q_image = QImage(frame_array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
        pixmap = QPixmap.fromImage(q_image)
        
        # Escalar el pixmap para que quepa en la etiqueta, manteniendo aspecto
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
        )

    @Slot(dict)
    def update_counts(self, counts):
        """Actualiza las etiquetas de conteo en la pestaña de Métricas."""
        total = 0
        
        car_count = counts.get("car", 0) # Usar las clases que definimos
        truck_count = counts.get("truck", 0)
        bus_count = counts.get("bus", 0)
        person_count = counts.get("person", 0)
        
        self.lbl_count_car.setText(str(car_count))
        self.lbl_count_truck.setText(str(truck_count))
        self.lbl_count_bus.setText(str(bus_count))
        self.lbl_count_person.setText(str(person_count))
        
        total = car_count + truck_count + bus_count + person_count
        self.lbl_count_total.setText(f"<strong>{total}</strong>")

    def closeEvent(self, event):
        """Nos aseguramos de detener el hilo de trabajo al cerrar la ventana."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker.stop_processing()
            self.worker_thread.quit()
            self.worker_thread.wait(5000) # Esperar max 5 seg
        event.accept()
