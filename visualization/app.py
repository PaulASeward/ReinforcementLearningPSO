import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import environment.functions as functions

# Load data
# There is typically 20 episodes with 100 timesteps, 10 particles, and 2 dimensions
# We will use the first episode for this example
positions = np.load('data/swarm_locations.npy')  # Shape is  (episodes, time_steps, particles, dimensions)
positions = positions[0]  # Shape is (time_steps, particles, dimensions)
valuations = np.load('data/swarm_evaluations.npy')  # Shape is (time_steps, particles)
valuations = valuations[0]  # Shape is (time_steps, particles)

# Evaluation function (adjust as necessary)
obj_f = functions.CEC_functions(dim=2, fun_num=6)
def eval(X):
    return obj_f.Y_matrix(np.array(X).astype(float))


# Create a meshgrid for the background surface
x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x, y)
points = np.stack([X.ravel(), Y.ravel()], axis=-1)
Z = eval(points).reshape(X.shape)


app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Graph(id='3d-swarm-visualization'),
    html.Button('Previous', id='btn-previous', n_clicks=0),
    html.Button('Next', id='btn-next', n_clicks=0),
    html.Button('Play', id='btn-play', n_clicks=0),
    html.Button('Stop', id='btn-stop', n_clicks=0),
    dcc.Slider(
        id='timestep-slider',
        min=0,
        max=len(positions) - 1,
        value=0,
        marks={i: str(i) for i in range(0, len(positions), 10)},
        step=1,
    ),
    dcc.Interval(
        id='auto-stepper',
        interval=1000, # in milliseconds
        n_intervals=0,
        disabled=True, # Start disabled
    )
])

@app.callback(
    Output('timestep-slider', 'value'),
    [Input('btn-previous', 'n_clicks'),
     Input('btn-next', 'n_clicks'),
     Input('btn-play', 'n_clicks'),
     Input('btn-stop', 'n_clicks'),
     Input('auto-stepper', 'n_intervals')],
    [State('timestep-slider', 'value')]
)
def update_slider(btn_previous, btn_next, btn_play, btn_stop, n_intervals, current_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-previous' and current_value > 0:
        return current_value - 1
    elif button_id == 'btn-next' and current_value < len(positions) - 1:
        return current_value + 1
    elif button_id == 'auto-stepper':
        if current_value < len(positions) - 1:
            return current_value + 1
        else:
            return 0  # Loop back to start, adjust as needed
    # Play and Stop buttons don't directly update the slider, so we ignore them here

    return dash.no_update

@app.callback(
    Output('auto-stepper', 'disabled'),
    [Input('btn-play', 'n_clicks'),
     Input('btn-stop', 'n_clicks')],
    [State('auto-stepper', 'disabled')]
)
def control_auto_stepper(btn_play, btn_stop, is_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_disabled

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'btn-play':
        return False
    elif button_id == 'btn-stop':
        return True

    return is_disabled


@app.callback(
    Output('3d-swarm-visualization', 'figure'),
    [Input('timestep-slider', 'value')]
)
def update_figure(selected_timestep):
    fig = go.Figure()

    # Color particles
    color_scale = px.colors.qualitative.Plotly
    for i in range(positions.shape[1]):  # Loop over particles
        current_positions = positions[selected_timestep, i, :]
        x = [current_positions[0]]
        y = [current_positions[1]]
        z = [valuations[selected_timestep, i]]

        # Add trace for each particle
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color=color_scale[i % len(color_scale)]),
            name=f'Particle {i + 1}'
        ))

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.7, cmin=np.min(Z), cmax=1000))

    # Layout adjustments
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        ),
        scene=dict(
            xaxis=dict(nticks=4, range=[-100, 100]),
            yaxis=dict(nticks=4, range=[-100, 100]),
            zaxis=dict(nticks=4, range=[np.min(Z), 1000]),
        ),
        autosize=False,
        width=1200,
        height=800
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
