import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def load_stylesheet(app):
    """Carga la hoja de estilo QSS."""
    try:
        with open("ui/theme.qss", "r") as f:
            style = f.read()
            app.setStyleSheet(style)
    except FileNotFoundError:
        print("Advertencia: No se encontró el archivo 'ui/theme.qss'.")

if __name__ == "__main__":
    # Importar nuestro motor C++ compilado
    try:
        import motor_contador
    except ImportError:
        print("Error: No se pudo encontrar el módulo 'motor_contador'.")
        print("Asegúrate de que el módulo C++ esté compilado y en la ruta.")
        sys.exit(1)

    app = QApplication(sys.argv)
    
    # --- AÑADIR ESTAS LÍNEAS ---
    load_stylesheet(app)
    # ---------------------------
    
    # Pasamos el módulo del motor a nuestra ventana principal
    window = MainWindow(motor_contador)
    window.show()
    
    sys.exit(app.exec())
