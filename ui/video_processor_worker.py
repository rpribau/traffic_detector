from PySide6.QtCore import QObject, Signal, Slot
import numpy as np

class VideoProcessorWorker(QObject):
    # --- Señales que este worker emitirá ---
    
    # Emite el frame de video procesado (como un array numpy)
    new_frame_ready = Signal(np.ndarray)
    
    # Emite los conteos actualizados (como un diccionario)
    new_counts_ready = Signal(dict)
    
    # Emite un mensaje de estado
    status_updated = Signal(str)
    
    # Emite cuando el procesamiento termina
    finished = Signal()

    def __init__(self, motor_module):
        super().__init__()
        self.motor_module = motor_module
        self.vehicle_counter = None
        self.is_running = False

    # --- Slots (funciones que la UI puede llamar) ---

    @Slot(str, str)
    def start_processing(self, video_path, model_path):
        """
        Inicia el procesamiento en este hilo.
        """
        try:
            # Inicializa el motor C++ *dentro de este hilo*
            self.vehicle_counter = self.motor_module.VehicleCounter(model_path)
            
            # (Asumimos que el motor C++ tiene un método para abrir el video)
            # Idealmente, el motor C++ maneja su propio bucle de video
            # y nosotros solo lo sondeamos.
            
            # --- Este es un ejemplo de bucle ---
            # (En una implementación real, tu VehicleCounter.start_processing 
            #  probablemente bloquearía y manejaría su propio bucle)
            
            self.status_updated.emit(f"Abriendo video: {video_path}")
            self.is_running = True

            # (Simulación de un bucle de video)
            # En tu implementación C++, esto sería un bucle real de OpenCV
            while self.is_running:
                # 1. Obtener datos del motor C++
                frame = self.vehicle_counter.get_latest_frame() # Debe devolver un np.array
                counts = self.vehicle_counter.get_counts()       # Debe devolver un dict
                
                if frame is None:
                    self.status_updated.emit("Video finalizado.")
                    break
                    
                # 2. Emitir señales a la UI
                self.new_frame_ready.emit(frame)
                self.new_counts_ready.emit(counts)
                
                # (Pequeña pausa para no saturar; en C++ esto estaría
                #  limitado por el framerate del video)
                # QThread.msleep(33) # ~30 FPS

        except Exception as e:
            self.status_updated.emit(f"Error en el motor: {e}")
        finally:
            self.finished.emit()

    @Slot()
    def stop_processing(self):
        """
        Detiene el bucle de procesamiento.
        """
        self.status_updated.emit("Deteniendo procesamiento...")
        self.is_running = False
        if self.vehicle_counter:
            self.vehicle_counter.stop_processing() # Decirle al motor C++ que pare