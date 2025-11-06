from imports_constants_paramaters import *
from simulation_analysis import selective_alleles_check, samples_pca
def generate_population(min_population_size, max_population_size, base_freqs):
    """
    Generates a single population with allele frequencies based on a normal distribution
    around base frequencies.

    Parameters:
        min_population_size (int): Minimum population size
        max_population_size (int): Maximum population size
        base_freqs (list of float): Base allele frequencies to sample from

    Returns:
        tuple: (population size, list of allele frequencies)
    """
    pop_size = np.random.randint(min_population_size, max_population_size + 1) if POP_SIZE is None else POP_SIZE
    freqs = []
    for i in range(len(base_freqs)):
        freq = np.random.normal(base_freqs[i], sigma)  # Sample from normal dist
        freq = max(0, min(freq, 1))  # Clamp between [0,1]
        freqs.append(freq)
    return pop_size, freqs


def generate_populations(num_of_populations, min_population_size, max_population_size, num_of_variants):
    """
    Initializes populations with base allele frequencies drawn from a beta distribution.

    Parameters:
        num_of_populations (int): Number of populations
        min_population_size (int): Minimum size of any population
        max_population_size (int): Maximum size of any population
        num_of_variants (int): Number of loci (variants)

    Returns:
        list: List of population data structures
    """
    populations = []
    base_freqs = [np.random.beta(0.3, 0.3) for _ in range(num_of_variants)]  # High variance in allele frequencies

    for i in range(num_of_populations):
        pop_size, freqs = generate_population(min_population_size, max_population_size, base_freqs)
        # Each population stores its entire simulation state
        populations.append([[pop_size], [freqs], pop_size, freqs, 0, areas[i]])
    pop_size, freqs = generate_population(min_population_size, max_population_size, base_freqs)
    populations.append([[pop_size], [freqs], pop_size, freqs, 0, 'unsample'])

    return populations


def migrate(populations, source_pop, target_pop, migrate_size):
    """
    Migrates a proportion of individuals from one population to another.

    Parameters:
        populations (list): List of population data structures
        source_pop (int): Index of source population
        target_pop (int): Index of target population
        migrate_size (float): Fraction of source population migrating
    """
    mig_pop_size = int(populations[source_pop][CUR_SIZE] * migrate_size)
    if mig_pop_size == 0:
        return

    # Create allele freqs for migrant individuals (no mutation)
    mig_freqs = generate_next_generation_freq(populations[source_pop][CUR_FREQS], mig_pop_size, False)

    tgt_pop_size = populations[target_pop][CUR_SIZE] + mig_pop_size

    # Recalculate allele frequencies in target population after migration
    tgt_counts = (
            populations[target_pop][CUR_SIZE] * np.array(populations[target_pop][CUR_FREQS])
            + mig_pop_size * mig_freqs
    )
    populations[target_pop][CUR_FREQS] = tgt_counts / tgt_pop_size


def split(populations, split_pop, split_gen):
    """
    Creates a new population by splitting an existing one at a specific generation.

    Parameters:
        populations (list): Existing populations
        split_pop (int): Index of the population to split
        split_gen (int): Generation of the split
    """
    split_pop_size = populations[split_pop][CUR_SIZE]
    split_freqs = populations[split_pop][CUR_FREQS]

    # Add a new population with identical properties
    populations.append([
        [split_pop_size],
        [split_freqs],
        split_pop_size,
        split_freqs,
        split_gen,
        populations[split_pop][AREA]
    ])


def replace(populations, old_pop, new_pop):
    """
    Replaces the allele frequencies in one population with those from another.

    Parameters:
        populations (list): List of population data structures
        old_pop (int): Index of the source population
        new_pop (int): Index of the population to be replaced
    """
    populations[new_pop][CUR_FREQS] = populations[old_pop][CUR_FREQS]


def generate_next_generation_freq(freqs, population_size, mutations):
    """
    Simulates allele frequency drift using binomial sampling (with optional mutation).

    Parameters:
        freqs (list): List of current allele frequencies
        population_size (int): Number of diploid individuals
        mutations (bool): Whether to apply mutation

    Returns:
        numpy.ndarray: Updated allele frequencies for next generation
    """
    new_freqs = np.zeros_like(freqs)

    for i in range(len(freqs)):
        p = freqs[i]

        if np.isnan(p) or p < 0 or p > 1:
            print(f"Invalid p: {p}, index: {i}, all freqs: {freqs}")
            raise ValueError("Invalid allele frequency detected")

        count = np.random.multinomial(2 * population_size, [p, 1 - p])
        new_freq = count[0] / (2 * population_size)

        if mutations:
            # Symmetric mutation model
            new_freq = new_freq * (1 - mutation_rate) + (1 - new_freq) * mutation_rate

        new_freqs[i] = new_freq

    return new_freqs


def genetic_drift_simulation(generations, populations, migrations, splits, replacements, mutations=False,
                             unsampled_pop_idx=None):
    """
    Simulates genetic drift over multiple generations with optional migration, population split, and replacement events.

    Parameters:
    -----------
    generations : int
        Total number of generations to simulate.

    populations : list
        A list of populations, each represented as a list containing:
        [0] SIZES - historical sizes over time,
        [1] FREQS - historical allele frequencies over time,
        [2] CUR_SIZE - current population size,
        [3] CUR_FREQS - current allele frequencies,
        [4] START_GEN - generation the population was created,
        [5] AREA - geographic label or area.

    migrations : list of lists
        Each migration event is a list containing:
        [0] SRC_POP - source population index,
        [1] TGT_POP - target population index,
        [2] MIG_GEN - generation at which migration starts,
        [3] MIG_SIZE - fraction of source population that migrates,
        [4] MIG_LENGTH - number of generations over which migration occurs.

    splits : list of lists
        Each split event is a list containing:
        [0] SRC_POP - index of the population to split from,
        [1] SPLIT_GEN - generation at which the split occurs.

    replacements : list of lists
        Each replacement event is a list containing:
        [0] SRC_POP - population providing new alleles,
        [1] TGT_POP - population being replaced,
        [2] RPC_GEN - generation at which replacement occurs.

    mutations : bool, optional (default=False)
        If True, introduces mutations during drift.

    Returns:
    --------
    tuple:
        - populations : list
            Updated list of populations after simulation.
        - result_matrix : np.ndarray
            Matrix containing allele frequencies for all populations at each generation.
            Each row: [population_index, generation, allele_freq_1, ..., allele_freq_n]
    """
    result_matrix = []
    baseline_migrations = []
    gen = 0
    for i in range(100):
        src_pop = np.random.randint(0, len(populations))
        tgt_pop = np.random.randint(0, len(populations))
        while tgt_pop == src_pop:
            tgt_pop = np.random.randint(0, len(populations))
        size = np.random.uniform(0.01, 0.05)
        duration = 1
        baseline_migrations.append([src_pop, tgt_pop, gen, size, duration])
        gen += 4

    for gen in range(generations):

        # migrations
        for migration in migrations:
            if gen < migration[MIG_GEN] + migration[MIG_LENGTH] and gen >= migration[MIG_GEN]:
                migrate(populations, migration[SRC_POP], migration[TGT_POP],
                        migration[MIG_SIZE] / migration[MIG_LENGTH])

        for migration in baseline_migrations:
            if gen < migration[MIG_GEN] + migration[MIG_LENGTH] and gen >= migration[MIG_GEN]:
                migrate(populations, migration[SRC_POP], migration[TGT_POP],
                        migration[MIG_SIZE] / migration[MIG_LENGTH])

        if unsampled_pop_idx != None:
            if np.random.rand() < p_mig_unsample:
                migrate(populations, unsampled_pop_idx, np.random.randint(0, 5), 0.01)

        # splits
        for splt in splits:
            if splt[SPLIT_GEN] == gen:
                split(populations, splt[SRC_POP], splt[SPLIT_GEN])

        # Replacement
        for replacement in replacements:
            if replacement[RPC_GEN] == gen:
                replace(populations, replacement[SRC_POP], replacement[TGT_POP])

        # drift
        for population in populations:
            if population[CUR_SIZE] > 0:
                population_num = populations.index(population)
                matrix_row = np.concatenate([[population_num, gen], population[CUR_FREQS]])
                result_matrix.append(matrix_row)

                cur_freqs = generate_next_generation_freq(population[CUR_FREQS], population[CUR_SIZE], mutations)
                population[CUR_FREQS] = cur_freqs
                population[FREQS].append(cur_freqs)
                population[CUR_SIZE] *= pop_growth_factor
                population[CUR_SIZE] = int(population[CUR_SIZE])
                population[SIZES].append(population[CUR_SIZE])

    result_matrix = np.array(result_matrix)
    return populations, result_matrix


def determine_number_of_samples(num_of_samples, populations):
    """
    Determine how many samples to draw from each population based on area weights,
    while ensuring the exact number of samples per area and per population matches the total.

    Args:
        num_of_samples (int): Total number of samples to distribute.
        populations (list): List of population dictionaries. Each must have an 'area' key.

    Returns:
        np.array: Number of samples per population (in the same order as input).
    """
    area_weights = {
        's_europe': 3.3,
        'britain': 1.2,
        'asia': 2.2,
        'levant': 1.7,
        'america': 1.6,
        'step': 1.2,
        'unsample': 0  # This area should not receive any samples
    }

    pop_num = len(populations)
    pop_num_of_samples = np.zeros(pop_num, dtype=int)

    # Step 1: Get the list of actual areas present in the population data (excluding 'unsample')
    actual_areas = sorted({pop[AREA] for pop in populations if pop[AREA] != 'unsample'})

    # Step 2: Extract weights only for the areas that are in use
    weights = np.array([area_weights[area] for area in actual_areas])
    norm_weights = weights / weights.sum()

    # Step 3: Determine how many samples to assign to each area based on the weights
    area_sample_counts = np.random.multinomial(num_of_samples, norm_weights)

    # Step 4: For each area, divide its samples among the populations within that area
    for area, area_count in zip(actual_areas, area_sample_counts):
        # Find indices of populations that belong to the current area
        pop_indices = [i for i, pop in enumerate(populations) if pop[AREA] == area]

        if not pop_indices:
            continue

        # Assign equal probability to each population in the area
        prob = np.ones(len(pop_indices)) / len(pop_indices)

        # Distribute samples using multinomial to ensure the total is exactly area_count
        samples_per_pop = np.random.multinomial(area_count, prob)

        # Update the sample count for each population
        for idx, pop_idx in enumerate(pop_indices):
            pop_num_of_samples[pop_idx] = samples_per_pop[idx]

    # print(pop_num_of_samples)
    return pop_num_of_samples



def determine_sample_times(population, num_of_samples):
    """
    Determine sampling times for a given population based on its area-specific time weights.

    Args:
        population (dict): Population dictionary containing AREA and other info.
        num_of_samples (int): Number of samples to draw.

    Returns:
        list: List of sampling generation times.
    """
    sample_times = []
    area = population[AREA]
    area_time_weights = areas_time_weights[area]
    for i in range(num_of_samples):
        time_val = np.random.rand()
        # Apply stay probability
        if i > 0 and np.random.rand() < p_stay:
            sample_times.append(sample_times[-1])
            continue
        # Resample if below minimum threshold
        while time_val < area_time_weights[0]:
            time_val = np.random.rand()
        for j in range(len(area_time_weights)):
            if time_val <= area_time_weights[j]:
                # Calculate time based on generation index and random offset
                time = generations - (4 * j + np.random.randint(0, 4))
                sample_times.append(time)
                break
    return sample_times


def generate_pop_samples(populations, population_idx, sample_times):
    """
    Generate genetic samples for a single population at specified sampling times.

    Args:
        populations (list): List of population dictionaries.
        population_idx (int): Index of population to sample from.
        sample_times (list): List of generation times for sampling.

    Returns:
        list: Samples containing population index, generation, and genotype data.
    """
    population = populations[population_idx]
    start_gen = population[START_GEN]
    freqs = population[FREQS]
    samples = []
    for time in sample_times:
        # Skip if sample time is before population start
        if time < start_gen:
            continue
        sample = [population_idx, time]
        # Generate genotype for each allele based on frequency and diploid assumption
        for freq in freqs[time - (start_gen + 1)]:
            rand = np.random.uniform(0, 1)
            sample.append(1 if rand < freq else 0)
            rand = np.random.uniform(0, 1)  # diploid second allele
            sample[-1] += 1 if rand < freq else 0
        samples.append(sample)
    return samples


def generate_samples(populations):
    """
    Generate genetic samples across all populations based on determined sample sizes and times.

    Args:
        populations (list): List of population dictionaries.

    Returns:
        list: Combined list of samples from all populations.
    """
    samples = []
    samples_num_list = determine_number_of_samples(num_of_samples, populations)
    for idx, population in enumerate(populations):
        if population[AREA] == 'unsample':
            continue
        pop_num_of_samples = int(samples_num_list[idx])
        sample_times = determine_sample_times(population, pop_num_of_samples)
        pop_samples = generate_pop_samples(populations, idx, sample_times)
        for sample in pop_samples:
            samples.append(sample)
    return samples


def generate_kmeans_data(num_of_populations, migrations, splits, replacements, mutations=True):
    # Step 1: Create initial populations
    populations = generate_populations(num_of_populations, min_population_size,
                                       max_population_size, num_of_variants)

    # Step 2: Run short simulation to stabilize variation
    populations, _ = genetic_drift_simulation(100, populations, [], init_splits, [], mutations)

    # Step 3: Truncate frequency history to final state, reset time
    for population in populations:
        population[FREQS] = population[FREQS][-1:]
        population[START_GEN] = 0

    last_pop = populations[-1]
    unsampled_pop = populations[0]
    for i in range(len(populations)):
        if populations[i][AREA] == 'unsample':
            unsampled_pop = populations[i]
            populations[i] = last_pop
            populations[-1] = unsampled_pop
            break

    # Step 4: Run main simulation with demographic events
    populations, res_matrix = genetic_drift_simulation(
        generations, populations, migrations, splits, replacements, False
    )

    # Step 5: sample the populations and run pca
    selective_alleles = selective_alleles_check(populations)
    samples = generate_samples(populations)
    df, explained_variance = samples_pca(populations, samples, selective_alleles)
    # plot_sample_pca(df, populations, migrations, splits, replacements, explained_variance)

    # step 6: arrange the data
    # Extract PC1 and PC2 from the dataframe
    X_pca = df[['PC1', 'PC2', 'PC3']].to_numpy()

    # Convert generation to years before present (assuming 25 years per generation)
    years_before_present = (generations - df['generation']) * 25
    dates_array = years_before_present.to_numpy()

    # Encode population labels as numeric IDs
    population_ids_array, uniques = pd.factorize(df['population'])

    # Calculate the sample timestep within each population (index of sample in that population)
    population_timesteps_array = df.groupby('population').cumcount().to_numpy()

    # Combine PCA components and temporal data into a single input array
    X = np.column_stack((X_pca, dates_array))

    return populations, X, dates_array, population_ids_array, population_timesteps_array, explained_variance[:3]

