import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import environment.functions as functions  # Make sure to adjust this import based on your project structure

# Load data (adjust paths as necessary)
positions = np.load('data/swarm_locations.npy')[0]  # Shape: (time_steps, particles, dimensions)
valuations = np.load('data/swarm_evaluations.npy')[0]  # Shape: (time_steps, particles)

# Evaluation function (adjust as necessary)
obj_f = functions.CEC_functions(dim=2, fun_num=6)


def eval(X):
    return obj_f.Y_matrix(np.array(X).astype(float))


# Dash app setup
app = dash.Dash(__name__)

# Create a meshgrid for the background surface
x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x, y)
points = np.stack([X.ravel(), Y.ravel()], axis=-1)
Z = eval(points).reshape(X.shape)

# App layout
app.layout = html.Div([
    dcc.Graph(id='3d-swarm-visualization'),
    html.Button('Previous', id='btn-previous', n_clicks=0),
    html.Button('Next', id='btn-next', n_clicks=0),
    dcc.Slider(
        id='timestep-slider',
        min=0,
        max=len(positions) - 1,
        value=0,
        marks={i: str(i) for i in range(0, len(positions), 10)},
        step=1,
    )
])

@app.callback(
    Output('timestep-slider', 'value'),
    [Input('btn-previous', 'n_clicks'),
     Input('btn-next', 'n_clicks')],
    [State('timestep-slider', 'value')]
)
def update_slider(btn_previous, btn_next, current_value):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-previous' and current_value > 0:
        return current_value - 1
    elif button_id == 'btn-next' and current_value < len(positions) - 1:
        return current_value + 1
    return current_value


@app.callback(
    Output('3d-swarm-visualization', 'figure'),
    [Input('timestep-slider', 'value')]
)
def update_figure(selected_timestep):
    current_positions = positions[selected_timestep]
    x = current_positions[:, 0]
    y = current_positions[:, 1]
    z = valuations[selected_timestep]

    # Create traces
    trace1 = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color='red')
    )

    trace2 = go.Surface(z=Z, x=x, y=y, colorscale='Viridis', opacity=0.7)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-100, 100]),
            yaxis=dict(nticks=4, range=[-100, 100]),
            zaxis=dict(nticks=4, range=[np.min(Z), np.max(Z)]),
        )
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
