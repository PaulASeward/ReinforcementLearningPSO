import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import matplotlib.colors as mcolors  # Import matplotlib colors
import environment.functions as functions
import math

# Evaluation function (adjust as necessary)
obj_f = functions.CEC_functions(dim=2, fun_num=6)
def eval(X):
    return obj_f.Y_matrix(np.array(X).astype(float))


def calculate_current_min_explored(positions, valuations):
    # Calculate the minimum value explored by the swarm and store the position and value at each time step.
    X1_min = None
    X2_min = None
    min_valuation = None

    min_explored = []

    for t in range(len(valuations)):
        current_positions = positions[t]
        current_valuations = valuations[t]
        min_valuation_t = np.min(current_valuations)

        if min_valuation is None or min_valuation_t < min_valuation:
            min_valuation = min_valuation_t
            min_valuation_index = np.argmin(current_valuations)
            X1_min = current_positions[min_valuation_index, 0]
            X2_min = current_positions[min_valuation_index, 1]

        min_explored.append([X1_min, X2_min, min_valuation])

    return min_explored


def get_distinct_colors(n):
    """Generate n distinct colors, using a cycling method if n exceeds the base palette size."""
    base_colors = px.colors.qualitative.Dark24  # This is a palette of dark colors
    if n <= len(base_colors):
        return base_colors[:n]
    else:
        # Extend the color palette by repeating and modifying slightly
        colors = []
        cycle_count = int(np.ceil(n / len(base_colors)))
        for i in range(cycle_count):
            for color in base_colors:
                modified_color = lighten_color(color, amount=0.1 * i)
                colors.append(modified_color)
                if len(colors) == n:
                    return colors
    return colors

def lighten_color(color, amount=0.5):
    """Lighten color by a given amount. Amount > 0 to lighten, < 0 to darken."""
    try:
        c = mcolors.to_rgb(color)
        c = mcolors.rgb_to_hsv(c)
        c = (c[0], c[1], max(0, min(1, c[2] * (1 + amount))))
        c = mcolors.hsv_to_rgb(c)
        return mcolors.to_hex(c)
    except:
        print('Error: Invalid color: ', color)
        return color


def generate_nonlinear_marks(shift, range):
    linear_marks = {}
    exponential_ceiling = math.ceil(math.log10(range))
    linear_steps = (0, exponential_ceiling*100, 100)  # Linear steps for the slider ex) [0,100,200,300,400]

    print("Created Nonlinear Marks with Linear Steps: ", exponential_ceiling)
    for i, val in enumerate(linear_steps):
        nonlinear_label = (10 ** i) - shift
        linear_marks[val] = f'{nonlinear_label}'
        print(f'Linear Step, {i}: Linear Value: {val}, Nonlinear Label Value: {nonlinear_label}')

    return linear_marks


def linear_to_nonlinear_value(linear_value, shift):
    # Convert linear value to the original scale
    print(f"Converting Linear Value, {linear_value} to Nonlinear Value with Shift: ", shift)
    exponent = linear_value / 100
    nonlinear_value = (10 ** exponent) - shift
    print("Nonlinear Value: ", nonlinear_value)
    return int(nonlinear_value)


# Load data
# There is typically 20 episodes with 100 timesteps, 10 particles, and 2 dimensions
# We will use the first episode for this example
data_dir_f6 = 'data/f6/locations_at_step_250/'
data_dir = 'data/f6/locations_at_step_250/'
positions = np.load(data_dir + 'swarm_locations.npy')  # Shape is  (episodes, time_steps, particles, dimensions)
positions1 = positions[0]  # Shape is (time_steps, particles, dimensions)

velocities = np.load(data_dir + 'swarm_velocities.npy')  # Shape is (time_steps, particles, dimensions)
velocities1 = velocities[0]  # Shape is (time_steps, particles, dimensions)

swarm_best_positions = np.load(data_dir + 'swarm_best_locations.npy')  # Shape is (time_steps, particles, dimensions)
swarm_best_positions1 = swarm_best_positions[0]  # Shape is (time_steps, particles, dimensions)

valuations = np.load(data_dir + 'swarm_evaluations.npy')  # Shape is (time_steps, particles)
valuations1 = valuations[0]  # Shape is (time_steps, particles)

meta_data = np.genfromtxt(data_dir + 'meta_data.csv', delimiter=',', dtype=None, names=True, encoding='utf-8')
meta_data1 = meta_data[0]  # Meta data for the first episode

min_explored = calculate_current_min_explored(positions1, valuations1)

# Create a meshgrid for the background surface
x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x, y)
points = np.stack([X.ravel(), Y.ravel()], axis=-1)
Z = eval(points).reshape(X.shape)

# Calculate the z range, and shift the scale to positive values starting from 1
z_min, z_max = np.min(Z), np.max(Z)
z_min, z_max = int(z_min), int(z_max)
range = z_max - z_min
shift = abs(min(z_min, 0)) if z_min <= 0 else - abs(max(z_min, 0))  # Shift the scale to positive values starting from 1 so we can use non-linear scaling

app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Graph(id='3d-swarm-visualization'),
        ], style={'width': '85%', 'display': 'inline-block'}),
        html.Div([
            dcc.Slider(
                id='z-max-slider',
                min=z_min,
                max=z_max,
                step=1,
                value=z_min+1000,
                marks=generate_nonlinear_marks(shift, range),
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
    html.Button('Toggle Best Positions', id='btn-toggle-best', n_clicks=0),
    html.Div([
        dcc.Slider(
                id='timestep-slider',
                min=0,
                max=len(positions1) - 1,
                value=0,
                marks={i: str(i) for i in range(0, len(positions1), 10)},
                step=1,
            ),
    ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    html.Div([
        dcc.Dropdown(
            id='particle-selector',
            options=[{'label': f'Particle {i + 1}', 'value': i} for i in range(positions1.shape[1])],
            placeholder="Focus on Single Particle Behavior",
            multi=True,
            value=[i for i in range(positions1.shape[1])]
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
     Input('btn-toggle-best', 'n_clicks'),
     Input('z-max-slider', 'value')],
    [State('particle-selector', 'value')]  # Include the state of the particle selector
)
def update_figure(selected_timestep, toggle_best, slider_value, selected_particles):
    fig = go.Figure()

    # # Adjust z_max to the original scale
    z_max_value = linear_to_nonlinear_value(slider_value, shift)

    # Color particles
    show_best_positions = toggle_best % 2 == 1  # Toggle visibility based on odd/even number of clicks
    num_particles = positions1.shape[1]
    dark_colors = get_distinct_colors(num_particles)
    light_colors = [lighten_color(color, amount=0.5) for color in dark_colors]

    for i in selected_particles:  # Loop over particles
        current_positions = positions1[selected_timestep, i, :]
        best_positions = swarm_best_positions1[selected_timestep, i, :]

        color = dark_colors[i % len(dark_colors)]
        lighter_color = light_colors[i % len(light_colors)]

        x = [current_positions[0]]
        y = [current_positions[1]]
        z = [valuations1[selected_timestep, i]]

        # Add trace for each particle
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color=color),
            name=f'Particle {i + 1}'
        ))

        if show_best_positions:
            fig.add_trace(go.Scatter3d(
                x=[best_positions[0]], y=[best_positions[1]], z=eval([best_positions]),
                mode='markers', marker=dict(size=5, color=lighter_color), name=f'Particle {i + 1} Best'
            ))

    # Add Previous Current Minimum Explored
    if selected_timestep >= 1:
        min_explored_t_x1, min_explored_t_x2, min_explored_t_z = min_explored[selected_timestep-1]
        fig.add_trace(go.Scatter3d(
            x=[min_explored_t_x1], y=[min_explored_t_x2], z=[min_explored_t_z],
            mode='markers',
            marker=dict(size=5, color='black'),
            name='Current Swarm Min'
        ))

    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.7, cmin=np.min(Z), cmax=z_max_value))

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
            zaxis=dict(nticks=4, range=[np.min(Z), z_max_value]),
        ),
        autosize=False,
        width=1200,
        height=800
    )
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
    elif button_id == 'btn-next' and current_value < len(positions1) - 1:
        return current_value + 1
    elif button_id == 'auto-stepper':
        if current_value < len(positions1) - 1:
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
