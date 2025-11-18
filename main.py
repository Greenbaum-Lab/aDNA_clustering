from imports_constants_paramaters import *
from simulation import *
from simulation_analysis import *
from clustering_evaluation import *


def simulate_drift(migrations, splits, replacements, mutations=False):
    """
    Simulates genetic drift with optional migrations, splits, and replacements.

    Parameters:
        migrations (list): Migration events.
        splits (list): Population split events.
        replacements (list): Replacement events.
        mutations (bool): Whether to include mutations (default False).

    Returns:
        None. Displays plots and statistics on drift results.
    """
    # Step 1: Create initial populations
    populations = generate_populations(5, min_population_size,
                                       max_population_size, num_of_variants)
    unsampled_pop_idx = len(populations)-1

    # Step 2: Run short simulation to stabilize variation
    populations, _ = genetic_drift_simulation(100, populations, [], [], [], False)

    # Step 3: Truncate frequency history to final state, reset time
    for population in populations:
        population[FREQS] = population[FREQS][-1:]
        population[START_GEN] = 0


    # Step 4: Calculate and visualize Site Frequency Spectrum
    sfs = calculate_SFS(populations)
    show_sfs(sfs)

    # Step 5: Run main simulation with demographic events
    populations, res_matrix = genetic_drift_simulation(generations, populations, migrations, splits, replacements, mutations, unsampled_pop_idx)

    # Step 6: Identify nearly fixed (selective) alleles
    selective_alleles = selective_alleles_check(populations)
    print('Number of selective alleles: ', len(selective_alleles))

    # Step 7: Count fixed alleles in each population
    for i, population in enumerate(populations):
        eliminated_alleles = 0
        for freq in population[CUR_FREQS]:
            if freq < frequency_threshold or freq > 1 - frequency_threshold:
                eliminated_alleles += 1
        print(f'Population {i+1} eliminated alleles: {eliminated_alleles}')

    # Step 8: Run PCA on variant and sample matrices
    result_pca(populations, res_matrix, selective_alleles, migrations, splits, replacements, frequency_threshold)
    samples = generate_samples(populations)
    df, explained_variance = samples_pca(populations, samples, selective_alleles)
    plot_sample_pca_separately(df, populations, migrations, splits, replacements, explained_variance)

    # Step 9: Calculate and show FST matrices over time
    fst_matrices = create_FST_matrices(populations, fst_generations)
    show_FST_matrices(populations, fst_matrices)

    # Step 10: Plot heterozygosity trend
    plot_heterozygosity_over_time(populations)


def generate_simulation_data(seed=None):
    if seed is not None:
        np.random.seed(seed)
    migrations=[]
    gen=0
    for i in range (25):
        src_pop=np.random.randint(0,7)
        tgt_pop=np.random.randint(0,7)
        while tgt_pop==src_pop:
            tgt_pop=np.random.randint(0,7)
        size = np.random.uniform(0.01, 0.2)
        duration=1
        migrations.append([src_pop, tgt_pop, gen, size, duration])
        gen+=16



    return run_clustering_evaluation(5, migrations, [], [])

# simulate_drift([], [], [], True)
# run_clustering_evaluation(4, [], [], [])



def run_simulation(seed):
    return generate_simulation_data(seed)


def run_many_simulations(num_of_simulations, num_workers):
    scores_for_k = []
    scores_for_t = []
    subpopulations_numbers = []
    if __name__ == "__main__":
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(run_simulation, range(num_simulations)))

    for sim_scores_for_k, sim_scores_for_t, sim_number_of_subpopulations in results:
        scores_for_k.append(sim_scores_for_k)
        scores_for_t.append(sim_scores_for_t)
        subpopulations_numbers.append(sim_number_of_subpopulations)

    temporal_weight_values_for_t = [0, 0.1, 0.2, 0.5, 0.75, 1, 10]
    k_values_for_t = np.arange(1, 41, 1)

    temporal_weight_values_for_k = np.arange(0, 1.5, 0.05)
    k_values_for_k = [5, 10, 20, 30, 40, 100]

    thresholds = scores_for_k[0].keys()
    for th in thresholds:
        th_scores_for_t = [sim[th] for sim in scores_for_t]
        th_scores_for_t = np.array(th_scores_for_t)
        th_scores_for_k = [sim[th] for sim in scores_for_k]
        th_scores_for_k = np.array(th_scores_for_k)

        th_subpopulations_numbers = [sim[th] for sim in subpopulations_numbers]
        th_subpopulations_numbers = np.array(th_subpopulations_numbers)
        mean_num_of_subpops = th_subpopulations_numbers.mean()

        mean_t = th_scores_for_t.mean(axis=0)
        std_t = th_scores_for_t.std(axis=0)


        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_ylim([0, 1])
        ax.set_xlabel('Number of clusters (K)')
        ax.set_ylabel('ARI')

        for i, t_val in enumerate(temporal_weight_values_for_t):
            mean_vals = mean_t[i]
            std_vals = std_t[i]
            ax.plot(k_values_for_t, mean_vals, label=f't={t_val}')
            ax.fill_between(k_values_for_t, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)
        ax.set_title(f"ARI for various temporal weights - threshold = {th}")
        ax.text(
            0.02, 0.95,
            f"Avg number of populations:\n{mean_num_of_subpops:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
        ax.legend()
        plt.show()

        mean_k = th_scores_for_k.mean(axis=0)
        std_k = th_scores_for_k.std(axis=0)


        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_ylim([0, 1])
        ax.set_xlabel('Temporal weight')
        ax.set_ylabel('ARI')

        for i, k_val in enumerate(k_values_for_k):
            mean_vals = mean_k[i]
            std_vals = std_k[i]
            ax.plot(temporal_weight_values_for_k, mean_vals, label=f'k={k_val}')
            ax.fill_between(temporal_weight_values_for_k, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)

        ax.set_title(f"ARI for various k - threshold = {th}")
        ax.text(
            0.02, 0.95,
            f"Avg number of populations:\n{mean_num_of_subpops:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
        ax.legend()
        plt.show()


    create_mean_ari_grid_heatmap(scores_for_t, temporal_weight_values_for_t, k_values_for_t)

num_simulations = 100
num_workers=50
run_many_simulations(num_simulations, num_workers)
















