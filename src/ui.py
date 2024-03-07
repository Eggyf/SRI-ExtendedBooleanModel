from models_sri import *


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# Función de búsqueda ficticia
def search(query):
    # Aquí deberías poner tu lógica de búsqueda real
    # Por ahora, simplemente mostraremos el resultado en la salida
    print("Buscando:", query)
    # Supongamos que el resultado de la búsqueda es una lista de resultados
    results = ["Resultado 1", "Resultado 2", "Resultado 3"]
    return results


# Inicializa la aplicación Dash
app = dash.Dash(__name__)

# Define el diseño de la aplicación
app.layout = html.Div(
    [
        html.H1("Búsqueda Simple"),
        dcc.Input(id="search-input", type="text", placeholder="Introduce tu consulta"),
        html.Button("Buscar", id="search-button", n_clicks=0),
        html.Div(id="search-results"),
    ]
)


# Define la función de búsqueda al hacer clic en el botón
@app.callback(
    Output("search-results", "children"),
    [Input("search-button", "n_clicks")],
    [Input("search-input", "value")],
)
def update_search_results(n_clicks, query):
    if n_clicks > 0:
        results = search(query)
        return html.Ul([html.Li(result) for result in results])


if __name__ == "__main__":
    app.run_server(debug=True)
