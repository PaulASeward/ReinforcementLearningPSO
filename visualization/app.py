import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from swarm_simulator import SwarmSimulator

# Load data
swarm = SwarmSimulator(fun_num=6)

app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='function-selector',
            options=swarm.get_available_functions(),
            clearable=False,
            placeholder="Select Function"
        ),
    ], style={'margin-bottom': '20px'}),
    html.Div([
        dcc.Dropdown(
            id='step-selector',
            options=[],
            clearable=False,
            placeholder="Select Step in Experiment"
        ),
    ], style={'margin-bottom': '20px'}),
    html.Div([
        dcc.Dropdown(
            id='episode-selector',
            options=[],
            clearable=False,
            placeholder="Select Episode"
        ),
    ], style={'margin-bottom': '20px'}),
    # Place this where you define your app.layout
    html.Div(id='episode-metadata-display', style={'margin-top': '20px'}),
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
    html.Div([
        dcc.Slider(
                id='timestep-slider',
                min=0,
                max=len(swarm.ep_positions) - 1,
                value=0,
                marks={i: str(i) for i in range(0, len(swarm.ep_positions), 10)},
                step=1,
            ),
    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    dcc.Interval(
        id='auto-stepper',
        interval=500, # in milliseconds
        n_intervals=0,
        disabled=True, # Start disabled
    )
])


@app.callback(
    [
        Output('step-selector', 'options'),
        Output('episode-selector', 'options'),
        Output('z-max-slider', 'marks',),
        Output('z-max-slider', 'value'),
        Output('episode-metadata-display', 'children'),
    ],
    [
        Input('function-selector', 'value'),
        Input('step-selector', 'value'),
        Input('episode-selector', 'value'),
    ],
    [
        State('step-selector', 'value'),
    ]
)
def update_swarm(fun_num_input, step_input, ep_input, current_step):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'function-selector':
        swarm.set_function_number(fun_num_input)
        marks = swarm.surface.marks
        swarm.surface.clear_traces()
        swarm.surface.generate_surface()
        return swarm.get_available_steps(), [], marks, int(marks[200]), dash.no_update

    if triggered_id == 'step-selector':
        swarm.load_swarm_data_for_step(step=step_input)
        return dash.no_update, swarm.get_available_episodes(step_input), dash.no_update, dash.no_update, dash.no_update

    if triggered_id == 'episode-selector':
        swarm.load_swarm_data_for_episode(episode=ep_input)

        if ep_input is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Select an episode to view metadata."

        metadata = swarm.meta_data
        episode_row = metadata[ep_input]

        if metadata is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "No metadata is available for the selected episode."

        # Create a HTML table with metadata
        table_header = html.Thead(html.Tr([html.Th(col) for col in metadata.dtype.names]))
        table_body = html.Tbody([html.Tr([html.Td(episode_row[col]) for col in metadata.dtype.names])])
        table = html.Table([table_header, table_body], style={'width': '100%'})

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, table


@app.callback(
    Output('3d-swarm-visualization', 'figure'),
    [Input('timestep-slider', 'value'),
     Input('z-max-slider', 'value'),],
)
def update_figure(selected_timestep, slider_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'z-max-slider':
        swarm.surface.update_z_visible_max(slider_value)
        swarm.surface.clear_traces()
        fig = swarm.surface.generate_surface()
    else:
        fig = swarm.surface.get_surface()

    fig = swarm.surface.plot_particles(fig, swarm.num_particles, selected_timestep, swarm.ep_positions, swarm.ep_valuations, swarm.ep_swarm_best_positions, swarm.min_explored, swarm.dark_colors, swarm.light_colors)

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
