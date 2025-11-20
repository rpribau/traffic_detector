from PySide6.QtCore import QObject, Signal, Slot, QTimer, QThread
import numpy as np

class VideoProcessorWorker(QObject):
    # --- Señales que este worker emitirá ---
    new_frame_ready = Signal(np.ndarray)
    new_counts_ready = Signal(dict)
    status_updated = Signal(str)
    finished = Signal()

    def __init__(self, motor_module):
        super().__init__()
        self.motor_module = motor_module
        self.vehicle_counter = None
        self.is_running = False
        
        # --- (NUEVO) QTimer para sondeo ---
        # Usamos un timer para preguntar por nuevos frames
        # en lugar de un 'while true' que consume 100% de CPU.
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_updates)

    @Slot(str, str)
    def start_processing(self, video_path, model_path):
        """
        Inicia el procesamiento C++ y comienza el sondeo.
        """
        try:
            self.status_updated.emit("Inicializando motor C++...")
            self.vehicle_counter = self.motor_module.VehicleCounter(model_path)
            
            self.status_updated.emit(f"Abriendo video: {video_path}")
            
            # Iniciar el hilo de procesamiento de C++
            self.vehicle_counter.start_processing(video_path)
            
            self.is_running = True
            
            # --- (MODIFICADO) Iniciar el timer ---
            # 67ms es ~15 FPS, un buen punto de partida.
            self.poll_timer.start(67) 
            self.status_updated.emit("Procesamiento iniciado.")

        except Exception as e:
            self.status_updated.emit(f"Error al iniciar el motor: {e}")
            self.finished.emit()

    @Slot()
    def _poll_updates(self):
        """
        Esta función es llamada por el QTimer cada 33ms.
        """
        if not self.is_running:
            return

        try:
            # 1. Obtener datos del motor C++
            frame = self.vehicle_counter.get_latest_frame() # Devuelve np.array
            counts = self.vehicle_counter.get_counts()       # Devuelve dict
            
            # 2. Emitir señales a la UI
            if frame.size > 0:
                self.new_frame_ready.emit(frame)
            
            self.new_counts_ready.emit(counts)

        except Exception as e:
            self.status_updated.emit(f"Error en el bucle de sondeo: {e}")
            self.stop_processing() # Detener todo si hay un error

    @Slot()
    def stop_processing(self):
        """
        Detiene el sondeo y el hilo de C++.
        """
        if not self.is_running:
            return
            
        self.status_updated.emit("Deteniendo procesamiento...")
        self.is_running = False
        
        # --- (MODIFICADO) Detener el timer ---
        self.poll_timer.stop()
        
        if self.vehicle_counter:
            self.vehicle_counter.stop_processing() # Decirle al motor C++ que pare
            
        self.finished.emit()