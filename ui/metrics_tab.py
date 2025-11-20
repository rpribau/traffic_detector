import sys
import folium
import io
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PySide6.QtCore import Qt, QUrl, Slot, QDateTime
from PySide6.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QLabel, 
    QDateEdit,
    QFormLayout,
    QSpacerItem,
    QSizePolicy,
    QSplitter
)
from PySide6.QtWebEngineWidgets import QWebEngineView

# --- Coordenadas de Monterrey ---
MONTERREY_COORDS = [25.6866, -100.3161]

# --- Colores para Plotly ---
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG_COLOR = "#1A2E51"
PLOTLY_CARD_COLOR = "#244984"
PLOTLY_TEXT_COLOR = "#DDEEFC"
PLOTLY_GRID_COLOR = "#2654A7"
PLOTLY_COLORS = ["#4599EC", "#69B7F1", "#99D0F7", "#C3E2FA"]


class MetricsTab(QWidget):
    """
    Widget rediseñado con QSplitter para Mapa (Folium) y Gráficas (Plotly).
    """
    def __init__(self, camera_locations):
        super().__init__()
        self.camera_locations = camera_locations
        
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        
        self.map_widget = MapWidget(camera_locations)
        splitter.addWidget(self.map_widget)

        self.graphs_widget = GraphWidget()
        splitter.addWidget(self.graphs_widget)

        splitter.setStretchFactor(0, 3) 
        splitter.setStretchFactor(1, 2) 
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

    @Slot(str)
    def center_map_on(self, camera_name):
        if camera_name in self.camera_locations:
            # (ACTUALIZADO) Ahora 'camera_locations' es un dict de dicts
            camera_data = self.camera_locations[camera_name]
            location = camera_data["coords"]
            self.map_widget.center_on(location)

# --- WIDGET DEL MAPA (IZQUIERDA) ---
class MapWidget(QWidget):
    def __init__(self, camera_locations):
        super().__init__()
        self.camera_locations = camera_locations
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.web_view = QWebEngineView()
        self.generate_and_load_map()
        layout.addWidget(self.web_view)

    def generate_and_load_map(self):
        m = folium.Map(location=MONTERREY_COORDS, zoom_start=13, tiles="OpenStreetMap")
        
        # (ACTUALIZADO) Iteramos la nueva estructura de datos
        for name, info in self.camera_locations.items():
            coords = info["coords"]
            direction = info.get("direction", "N/A")
            
            # Mostramos Nombre + Dirección en el popup
            popup_html = f"<strong>{name}</strong><br>Dir: {direction}"
            
            folium.Marker(
                location=coords,
                popup=popup_html, 
                tooltip=name
            ).add_to(m)
        
        data = io.BytesIO()
        m.save(data, close_file=False)
        html_content = data.getvalue().decode()
        self.web_view.setHtml(html_content)

    def center_on(self, location):
        js_script = f"map.setView([{location[0]}, {location[1]}], 16);"
        self.web_view.page().runJavaScript(js_script)

# --- WIDGET DE GRÁFICAS (DERECHA) ---

class GraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("graph_widget_container")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10) 
        
        filter_layout = QFormLayout()
        filter_layout.setContentsMargins(0, 0, 0, 10) 
        
        self.date_filter = QDateEdit()
        self.date_filter.setCalendarPopup(True)
        self.date_filter.setDateTime(QDateTime.currentDateTime()) 
        
        filter_layout.addRow("Filtrar por Fecha:", self.date_filter)
        
        layout.addLayout(filter_layout) 
        
        self.plot_view = QWebEngineView()
        layout.addWidget(self.plot_view, 1)
        
        self.update_charts(["Carro", "Bus", "Camión"], [10, 5, 3]) 

    def update_charts(self, labels, data):
        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{"type": "bar"}], [{"type": "pie"}]],
            subplot_titles=("Conteo Total por Tipo", "Distribución de Vehículos"),
            vertical_spacing=0.15
        )

        fig.add_trace(
            go.Bar(x=labels, y=data, name="Conteo", marker_color=PLOTLY_COLORS[0]),
            row=1, col=1
        )

        fig.add_trace(
            go.Pie(
                labels=labels, 
                values=data, 
                name="Distribución", 
                hole=.4,
                marker_colors=PLOTLY_COLORS
            ),
            row=2, col=1
        )

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PLOTLY_BG_COLOR,
            plot_bgcolor=PLOTLY_CARD_COLOR,
            font_color=PLOTLY_TEXT_COLOR,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(font_color=PLOTLY_TEXT_COLOR),
            title_font_color=PLOTLY_TEXT_COLOR,
        )
        fig.update_yaxes(gridcolor=PLOTLY_GRID_COLOR, row=1, col=1)
        fig.update_annotations(font_color=PLOTLY_TEXT_COLOR)

        html_string = fig.to_html(
            full_html=True, 
            include_plotlyjs='cdn', 
            config={'displayModeBar': False}
        )
        
        html_string = html_string.replace(
            "<head>", 
            "<head><style>body, html { background-color: transparent; }</style>"
        )

        self.plot_view.setHtml(html_string)