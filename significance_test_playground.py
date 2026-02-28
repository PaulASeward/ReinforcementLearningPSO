from scipy import stats
from scipy.stats import t
import math

f6 = 23.7
f6_std_dev = 4.34

f11 = 87.6
f11_std_dev = 15.1

f14 = 2300
f14_std_dev = 380

f19 = 6.34
f19_std_dev = 1.30

f23 = 4720
f23_std_dev = 706

pso_std_devs = [
    f6_std_dev,
    f11_std_dev,
    f14_std_dev,
    f19_std_dev,
    f23_std_dev
]

mean_pso = [
    f6,
    f11,
    f14,
    f19,
    f23
]

mean_dqn = [
    45.0,
    20.8,
    1170,
    3.70,
    5740
]

mean_drqn = [
    42.4,
    18.1,
    1190,
    4.95,
    4890
]

# mean_ddpg = [
#     45.7,
#     23.4,
#     827,
#     4.63,
#     4520
# ]

mean_ddpg = [
    22.8,
    93.0,
    2430,
    5.67,
    4420
]

# std_dev_dqn = [
#     19.7,
#     10.2,
#     357,
#     1.46,
#     584
# ]
#
# std_dev_ddpg = [
#     13.9,
#     7.49,
#     342,
#     0.88,
#     667
# ]


def calculate_standard_two_sample_t_test(mean1, mean2, std_dev, n1, n2):
    """
    Perform a two-sample t-test assuming equal variances.

    Parameters:
    - mean1: Mean of sample 1
    - mean2: Mean of sample 2
    - std_dev: Common (pooled) standard deviation
    - n1: Sample size of sample 1
    - n2: Sample size of sample 2
    - alpha: Significance level (default = 0.05)

    Returns:
    - t_stat: The t statistic
    - df: Degrees of freedom
    - p_value: Two-tailed p-value
    - significant: True if difference is statistically significant
    """

    # Standard error of the difference
    se = std_dev * math.sqrt(1 / n1 + 1 / n2)

    # t statistic
    t_stat = (mean1 - mean2) / se

    # degrees of freedom for equal variance t-test
    df = n1 + n2 - 2

    # two-tailed p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))

    return t_stat, df, p_value

def calculate_standard_t_test(mean1,mean2,std_dev,n1,n2):
    # Standard error of the difference
    se = std_dev * math.sqrt(1 / n1 + 1 / n2)

    # t statistic
    t_stat = (mean1 - mean2) / se

    # degrees of freedom for equal variance t-test
    df = n1 + n2 - 2

    # two-tailed p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))

    return t_stat, p_value

def calculate_t_test(population_mean, population_std_dev, population_size, sample_mean, sample_std, sample_size):
    """
    Perform Welch's Independent t-test

    Parameters:
    population_mean (float): The mean of the population to compare against.
    population_std_dev (float): The standard deviation of the population.
    population_size (int): The size of the population.
    sample_mean (float): The mean of the sample.
    sample_std (float): The standard deviation of the sample.
    sample_size (int): The size of the sample.

    Returns:
    tuple: t-statistic and p-value
    """
    # Standard error
    se1 = population_std_dev ** 2 / population_size
    se2 = sample_std ** 2 / sample_size
    se = math.sqrt(se1 + se2)

    # t-statistic
    t_stat = (population_mean - sample_mean) / se

    # degrees of freedom (Welch–Satterthwaite equation)
    df_numerator = (se1 + se2) ** 2
    df_denominator = (se1 ** 2) / (population_size - 1) + (se2 ** 2) / (sample_size - 1)
    df = df_numerator / df_denominator

    # two-tailed p-value
    p_value = 2 * t.sf(abs(t_stat), df)

    return t_stat, p_value


def calculate_t_tests_for_group(group_population_means, group_population_std_devs, group_sample_means, group_std_devs):
    group_names = ['f6', 'f11', 'f14', 'f19', 'f23']
    group_size = 100
    sample_size = 200

    for population_mean, population_std_dev, mean, std_dev, name in zip(group_population_means,
                                                                        group_population_std_devs, group_sample_means,
                                                                        group_std_devs, group_names):
        # t_statistic, p_value = calculate_t_test(population_mean, population_std_dev, group_size, mean, std_dev, sample_size)
        # t_statistic, p_value =  (mean1=population_mean, mean2=mean, std_dev=population_std_dev, n1=group_size, n2=sample_size)
        t_statistic, p_value = calculate_standard_t_test(mean2=population_mean, mean1=mean, std_dev=population_std_dev, n1=group_size, n2=sample_size)
        print(f"T-test for {name}:")
        print(f"  T-statistic: {t_statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")


calculate_t_tests_for_group(mean_pso, pso_std_devs, mean_ddpg, pso_std_devs)

# group_names = ['f6', 'f11', 'f14', 'f19', 'f23']
# group_size = 30
# sample_size = 100
#
# for population_mean, population_std_dev, mean, name in zip(mean_pso, pso_std_devs, mean_ddpg, group_names):
#     # t_statistic, p_value = calculate_t_test(population_mean, population_std_dev, group_size, mean, std_dev, sample_size)
#     # t_statistic, p_value =  (mean1=population_mean, mean2=mean, std_dev=population_std_dev, n1=group_size, n2=sample_size)
#     t_statistic, p_value = calculate_standard_t_test(mean2=population_mean, mean1=mean, std_dev=population_std_dev, n1=group_size, n2=sample_size)
#     print(f"T-test for {name}:")
#     print(f"  T-statistic: {t_statistic:.4f}")
#     print(f"  P-value: {p_value:.4f}")
