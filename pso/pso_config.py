import copy
import os

class PSOConfig(object):
    # PSO Config:
    is_sub_swarm = False
    use_mock_data = False

    w = 0.729844  # Inertia weight
    c1 = 2.05 * w  # Social component Learning Factor
    c2 = 2.05 * w  # Cognitive component Learning Factor

    # w_min = 0.43  # Min of 5 decreases of 10%
    # w_max = 1.175  # Max of 5 increases of 10%
    w_min = 0.23  # Min of 5 decreases of 10%
    w_max = 1.375  # Max of 5 increases of 10%

    c_min = 0.883  # Min of 5 decreases of 10%
    c_max = 2.409  # Max of 5 increases of 10%

    # w_min = 0  # Min of 5 decreases of 10%
    # w_max = 1.44  # Max of 5 increases of 10%
    # c_min = 0  # Min of 5 decreases of 10%
    # c_max = 3.3 # Max of 5 increases of 10%
    rangeF = 100
    # v_min = 59.049
    # v_max = 161.051

    num_swarm_obs_intervals = 10
    swarm_obs_interval_length = 30

    v_min = 50
    v_max = 150

    v_min_scaling_factor = 0.5
    v_max_scaling_factor = 1.5

    velocity_braking = 1.0
    velocity_braking_min = 0.75
    velocity_braking_max = 1.25

    distance_threshold = 0.00
    distance_threshold_min = -0.20
    distance_threshold_max = 0.20

    replacement_threshold = 1.0
    replacement_threshold_min = 0.75
    replacement_threshold_max = 1.25

    # EXPERIMENT PARAMETERS
    fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
               -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
               700, 800, 900, 1000, 1100, 1200, 1300, 1400]

    best_f_standard_pso = [0,0,0,0,0,-26,-112,-21,-26,0,-85,-120,-200,-2279,-4080,-1,-104,-130,-5,-12,-305,-2513,-4594,-276,0,0,0,0]
    swarm_improvement_pso = [0,0,0,0,2,31,20,1,0,18,2,10,20,118,1300,1,43,141,4,1,1,237,1075,1,0,0,0,0]
    standard_deviations = [0.00e+00, 7.82e+05, 7.06e+07, 4.55e+03, 0.00e+00, 4.34e+00, 1.74e+01, 5.51e-2, 1.95e+00, 5.53e-2, 1.51e+01, 1.72e+01, 2.21e+01, 3.80e+02, 6.25e+02, 3.51e-1, 1.55e+01, 2.68e+01, 1.30e+00, 5.12e-1, 5.30e+01, 5.15e+02, 7.06e+02, 5.64e+00, 7.12e+00, 4.61e+01, 7.43e+01, 2.82e-13]
    standard_pso_results_dir = "pso/standard_pso_results"

    def __init__(self, func_num, swarm_algorithm, num_subswarms=None, use_mock_data=False, swarm_size=50, pso_dim=30):
        self.func_num = func_num
        self.pso_dim = pso_dim
        self.swarm_size = swarm_size
        self.swarm_algorithm = swarm_algorithm
        self.use_mock_data = use_mock_data
        self.standard_pso_path = os.path.join(self.standard_pso_results_dir, f"f{func_num}.csv")

        if num_subswarms is not None:
            if self.swarm_algorithm == "PMSO":
                self.num_sub_swarms = num_subswarms
            else:
                self.num_sub_swarms = 1

    def clone(self):
        return copy.deepcopy(self)
