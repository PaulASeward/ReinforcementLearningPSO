import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from swarm_simulator import SwarmSimulator

# Load data
swarm = SwarmSimulator(fun_num=6)


app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    dcc.Dropdown(
        id='function-selector',
        options=[{'label': f'Function {i + 1}', 'value': i} for i in range(29)],
        value=6,
        clearable=False,
        placeholder="Select Function"
    ),
    dcc.Dropdown(
        id='episode-selector',
        options=[{'label': f'Episode {i + 1}', 'value': i} for i in range(swarm.ep_positions.shape[0])],
        value=0,
        clearable=False,
        placeholder="Select Episode"
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='3d-swarm-visualization'),
        ], style={'width': '85%', 'display': 'inline-block'}),
        html.Div([
            dcc.Slider(
                id='z-max-slider',
                min=0,
                max=500,
                step=1,
                value=int(swarm.surface.marks[200]),
                marks=swarm.surface.marks,
                vertical=True,
                verticalHeight=400
            ),
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Button('Previous', id='btn-previous', n_clicks=0),
    html.Button('Next', id='btn-next', n_clicks=0),
    html.Button('Play', id='btn-play', n_clicks=0),
    html.Button('Stop', id='btn-stop', n_clicks=0),
    dcc.Dropdown(
        id='speed-selector',
        options=[
            {'label': '1x Speed', 'value': 500},
            {'label': '1.25x Speed', 'value': 400},
            {'label': '1.5x Speed', 'value': 333},
            {'label': '1.75x Speed', 'value': 285},
            {'label': '2x Speed', 'value': 250},
            {'label': '5x Speed', 'value': 100},
            {'label': '10x Speed', 'value': 50}
        ],
        value=500,  # Default to 1x speed
        clearable=False,
        placeholder="Select Playback Speed"
    ),
    dcc.Dropdown(
        id='particle-best-position',
        options=[
            {'label': 'Yes', 'value': True},
            {'label': 'No', 'value': False},
        ],
        value=False,
        clearable=False,
        placeholder="Display a Particle's Best Positions"
    ),
    html.Div([
        dcc.Slider(
                id='timestep-slider',
                min=0,
                max=len(swarm.ep_positions) - 1,
                value=0,
                marks={i: str(i) for i in range(0, len(swarm.ep_positions), 10)},
                step=1,
            ),
    ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        dcc.Dropdown(
            id='particle-selector',
            options=[{'label': f'Particle {i + 1}', 'value': i} for i in range(swarm.ep_positions.shape[1])],
            placeholder="Focus on Single Particle Behavior",
            multi=True,
            value=[i for i in range(swarm.ep_positions.shape[1])]
        ),
    ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    dcc.Interval(
        id='auto-stepper',
        interval=500, # in milliseconds
        n_intervals=0,
        disabled=True, # Start disabled
    )
])


@app.callback(
    Output('3d-swarm-visualization', 'figure'),
    [Input('timestep-slider', 'value'),
     Input('particle-best-position', 'value'),
     Input('z-max-slider', 'value'),
     Input('particle-selector', 'value')],
)
def update_figure(selected_timestep, show_p_best, slider_value, selected_particles):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'z-max-slider':
        swarm.surface.update_z_visible_max(slider_value)
        swarm.surface.clear_traces()
        fig = swarm.surface.generate_surface()
    elif button_id == 'particle-best-position':
        print("Particle best position changed")
        swarm.surface.clear_traces()
        fig = swarm.surface.generate_surface()
    elif button_id == 'particle-selector':
        print("Particle selector changed")
        swarm.surface.clear_traces()
        fig = swarm.surface.generate_surface()
    else:
        fig = swarm.surface.get_surface()

    fig = swarm.surface.plot_particles(fig, selected_particles, show_p_best, selected_timestep, swarm.ep_positions, swarm.ep_valuations, swarm.ep_swarm_best_positions, swarm.min_explored, swarm.dark_colors, swarm.light_colors)

    # return dict(data=fig.data, layout={'legend': {'uirevision': True}})
    return fig


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
    elif button_id == 'btn-next' and current_value < len(swarm.ep_positions) - 1:
        return current_value + 1
    elif button_id == 'auto-stepper':
        if current_value < len(swarm.ep_positions) - 1:
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
    Output('auto-stepper', 'interval'),
    [Input('speed-selector', 'value')]
)
def update_interval(speed):
    return speed

if __name__ == '__main__':
    app.run_server(debug=True)
