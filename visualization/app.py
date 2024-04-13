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
            value=6,
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
            value=0,
            clearable=False,
            placeholder="Select Episode"
        ),
    ], style={'margin-bottom': '20px'}),

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
        Output('3d-swarm-visualization', 'figure'),
        Output('step-selector', 'options'),
        Output('episode-selector', 'options'),
        Output('timestep-slider', 'max'),
        Output('timestep-slider', 'marks')
    ],
    [
        Input('function-selector', 'value'),
        Input('step-selector', 'value'),
        Input('episode-selector', 'value'),
        Input('timestep-slider', 'value'),
        Input('z-max-slider', 'value')
    ],
    [
        State('function-selector', 'value'),
        State('step-selector', 'value'),
        State('episode-selector', 'value'),
        State('3d-swarm-visualization', 'figure'),
        State('timestep-slider', 'value')
    ]
)
def update_figure(fun_num_input, step_input, ep_input, timestep_slider_value_input, z_max_slider_value_input, current_fun_num, current_step, current_ep, current_figure, current_timestep):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id in ['function-selector', 'step-selector', 'episode-selector']:
        print(f"Function: {fun_num_input}, Step: {step_input}, Episode: {ep_input}")
        swarm.set_function_number(fun_num_input)
        return dash.no_update, swarm.get_available_steps(), dash.no_update, dash.no_update, dash.no_update
    elif triggered_id in ['step-selector', 'episode-selector']:

        print(f"Function: {current_fun_num}, Step: {step_input}, Episode: {ep_input}")
        # Reload data for new function or step number
        swarm.load_completed_swarm_data(step=step_input, episode=0)

        # Check and reset the timestep if it is out of bounds
        max_timestep = len(swarm.ep_positions) - 1
        if current_timestep > max_timestep:
            current_timestep = 0

        # Update marks for the slider based on new data
        marks = {i: str(i) for i in range(0, len(swarm.ep_positions), 10)}

        # Regenerate the figure with new data
        fig = swarm.surface.generate_surface()
        fig = swarm.surface.plot_particles(fig, swarm.num_particles, current_timestep, swarm.ep_positions, swarm.ep_valuations, swarm.ep_swarm_best_positions, swarm.min_explored, swarm.dark_colors, swarm.light_colors)

        available_episodes = swarm.get_available_episodes(step_input) if step_input is not None else []
        print(f"Available Episodes: {available_episodes}")

        return fig, swarm.get_available_steps(), available_episodes, max_timestep, marks

    elif triggered_id in ['timestep-slider', 'z-max-slider']:
        print(f"Timestep: {timestep_slider_value_input}, Z-Max: {z_max_slider_value_input}")
        # Update visualization based on slider changes
        if triggered_id == 'z-max-slider':
            swarm.surface.update_z_visible_max(z_max_slider_value_input)
            swarm.surface.clear_traces()
            fig = swarm.surface.generate_surface()
        else:
            fig = swarm.surface.get_surface()

        fig = swarm.surface.plot_particles(fig, swarm.num_particles, timestep_slider_value_input, swarm.ep_positions, swarm.ep_valuations, swarm.ep_swarm_best_positions, swarm.min_explored, swarm.dark_colors, swarm.light_colors)

        return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    print(f"Unknown trigger: {triggered_id}")
    return current_figure, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# @app.callback(
#     Output('3d-swarm-visualization', 'figure'),
#     [Input('timestep-slider', 'value'),
#      Input('z-max-slider', 'value'),],
# )
# def update_figure(selected_timestep, slider_value):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         return dash.no_update
#
#     button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     if button_id == 'z-max-slider':
#         swarm.surface.update_z_visible_max(slider_value)
#         swarm.surface.clear_traces()
#         fig = swarm.surface.generate_surface()
#     else:
#         fig = swarm.surface.get_surface()
#
#     fig = swarm.surface.plot_particles(fig, swarm.num_particles, selected_timestep, swarm.ep_positions, swarm.ep_valuations, swarm.ep_swarm_best_positions, swarm.min_explored, swarm.dark_colors, swarm.light_colors)
#
#     return fig


# @app.callback(
#     [
#         Output('3d-swarm-visualization', 'figure'),
#         Output('timestep-slider', 'max'),
#         Output('timestep-slider', 'marks')
#     ],
#     [
#         Input('function-selector', 'value'),
#         Input('step-selector', 'value'),
#         Input('episode-selector', 'value')
#     ],
#     [
#         State('timestep-slider', 'value')
#     ]
# )
# def update_swarm(fun_num, step_num, ep, current_timestep):
#     # Update the swarm simulator with the new function number and step
#     global swarm
#     swarm = SwarmSimulator(fun_num=fun_num)
#     swarm.load_completed_swarm_data(step=step_num, episode=0)
#
#     # Reset timestep if the current timestep is out of new range
#     if current_timestep >= len(swarm.ep_positions):
#         current_timestep = 0
#
#     # Update the visualization
#     fig = swarm.surface.generate_surface()
#     fig = swarm.surface.plot_particles(fig, swarm.num_particles, current_timestep, swarm.ep_positions,
#                                        swarm.ep_valuations, swarm.ep_swarm_best_positions, swarm.min_explored,
#                                        swarm.dark_colors, swarm.light_colors)
#
#     # Update marks for the new data
#     marks = {i: str(i) for i in range(0, len(swarm.ep_positions), 10)}
#
#     return fig, len(swarm.ep_positions) - 1, marks

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
