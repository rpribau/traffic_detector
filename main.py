import sys
import os # <-- (NUEVO) Importar 'os'

# --- (NUEVO) Forzar el backend de Matplotlib ---
#
# 1. Le decimos a matplotlib que use 'pyside6' como su "API de Qt".
#    Esto debe hacerse ANTES de importar matplotlib.backends.
os.environ['QT_API'] = 'pyside6' 

# 2. Importamos matplotlib
import matplotlib

# 3. Usamos 'qtagg' (el backend genérico de Qt) en lugar de 'Qt6Agg'.
#    Gracias a la variable de entorno, 'qtagg' se conectará a PySide6.
matplotlib.use('qtagg') 
# ---------------------------------------------------

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
    
    # --- Cargar estilo ---
    load_stylesheet(app)
    
    # Pasamos el módulo del motor a nuestra ventana principal
    window = MainWindow(motor_contador)
    window.show()
    
    sys.exit(app.exec())