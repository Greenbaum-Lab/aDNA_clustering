from imports_constants_paramaters import *
from kmeans import *
from simulation_analysis import calculate_FST

def evaluate_clustering(clusters: np.ndarray, population_ids_array: np.ndarray, labels: np.ndarray):
    """
    Evaluate how well the clustering matches the original populations and temporal groups.

    Parameters:
    - clusters: np.ndarray, shape (n_samples,)
    - population_ids_array: np.ndarray, original population IDs
    - unique_population_timestep: np.ndarray, unique feature combining population and timestep
    """
    from sklearn.metrics import adjusted_rand_score
    if labels.dtype.kind in {'U', 'O'}:
        le = LabelEncoder()
        labels = le.fit_transform(labels)

    # Calculate Adjusted Rand Index (ARI) to evaluate clustering performance for temporal groups
    ari_score_population = adjusted_rand_score(labels, clusters)

    # print(f"Adjusted Rand Index (ARI) for Population and Temporal Groups: {ari_score_time:.2f}")
    return ari_score_population


def evaluate_migrations(populations, migrations,generations = 400, generation_time=25):
    """

    The score is based on:
    - D_i: genetic distance between source and target before migration
    - m_i: migration size

    Returns list of dicts per migration with delta_i, D_i, m_i, and final score.
    """
    migrations_data = []

    for i, migration in enumerate(migrations):
        src_pop = migration[SRC_POP]
        tgt_pop = migration[TGT_POP]
        mig_gen = migration[MIG_GEN]
        mig_size = migration[MIG_SIZE]
        mig_length = migration[MIG_LENGTH]

        # Translate generation to year
        mig_year = (generations - mig_gen) * generation_time
        mig_end_year = (generations - (mig_gen + mig_length)) * generation_time

        # D_i: genetic distance(FST) between source and target at time of migration
        D_i = calculate_FST(populations[src_pop], populations[tgt_pop], mig_gen)

        # Final score
        m_i = mig_size / 0.2
        D_i /= 0.2

        migrations_data.append({
            'migration_index': i,
            'D_i': D_i,
            'm_i': m_i,
            'src': src_pop,
            'tgt': tgt_pop,
            'year': mig_year
        })

    return migrations_data

def calculate_ARI_vs_t(X, temporal_weight_values, k_values, population_ids_array, event_stage_labels, explained_variance):
    scores_for_t = np.zeros((len(temporal_weight_values), len(k_values)))
    #Perform clustering and evaluate ARI scores
    for i_t, temporal_weight in enumerate(temporal_weight_values):
        for i_k, k in enumerate(k_values):
            print(f"temporal weight is {temporal_weight}, k is {k}")
            clusters, centroids = run_kmeans(X, k=k, temporal_weight=temporal_weight, genetic_weights=explained_variance)
            scores_for_t[i_t, i_k] = evaluate_clustering(clusters, population_ids_array, event_stage_labels)
    return scores_for_t

def calculate_ARI_vs_k(X, temporal_weight_values, k_values, population_ids_array, event_stage_labels, explained_variance):
    scores_for_k = np.zeros((len(k_values), len(temporal_weight_values)))
    for i_k, k in enumerate(k_values):
        for i_t, temporal_weight in enumerate(temporal_weight_values):
            print(f"k is {k}, temporal weight is {temporal_weight}")
            clusters, centroids = run_kmeans(X, k=k, temporal_weight=temporal_weight, genetic_weights=explained_variance)
            scores_for_k[i_k, i_t] = evaluate_clustering(clusters, population_ids_array, event_stage_labels)
    return scores_for_k


def run_clustering_evaluation(num_of_populations, migrations, splits, replacements):
    # Generate PCA and clustering data
    print("generating data")
    populations, X, dates_array, population_ids_array, population_timesteps_array, explained_variance = generate_kmeans_data(
        num_of_populations, migrations, splits, replacements)
    original_variance = explained_variance.copy()
    explained_variance /= explained_variance[0]

    # downsample
    X_down, dates_down, pop_ids_down, timesteps_down = downsample_within_populations(
        X, dates_array, population_ids_array, population_timesteps_array,
        max_samples_per_bin=50, bin_width=500
    )

    migrations_data = evaluate_migrations(populations, migrations)

    # plot_mig_scores_vs_size_and_distance(populations, 0.5, 25, migrations, X_down, pop_ids_down, dates_down, explained_variance)

    thresholds = [0, 0.05, 0.1, 0.15, 0.2]
    # plot_kmeans_colored_by_pop(X_down, dates_array=dates_down, pop_clusters=pop_ids_down, temporal_weight=0, genetic_weights=explained_variance, k=7)
    # plot_kmeans_colored_by_pop(X_down, dates_array=dates_down, pop_clusters=pop_ids_down, temporal_weight=0, genetic_weights=explained_variance, k=10)
    # plot_kmeans_colored_by_pop(X_down, dates_array=dates_down, pop_clusters=pop_ids_down, temporal_weight=0, genetic_weights=explained_variance, k=15)
    # plot_kmeans_colored_by_pop(X_down, dates_array=dates_down, pop_clusters=pop_ids_down, temporal_weight=0, genetic_weights=explained_variance, k=20)
    # plot_kmeans_colored_by_pop(X_down, dates_array=dates_down, pop_clusters=pop_ids_down, temporal_weight=0.75, genetic_weights=explained_variance, k=20)

    threshold_scores_for_k = {}
    threshold_scores_for_t = {}
    treshold_number_of_subpopulations = {}


    for th in thresholds:
        print(f"############# running evaluation for threshold - {th} #############")
        event_stage_labels = label_by_demographic_events(pop_ids_down, dates_down, migrations_data, splits,
                                                         replacements, threshold=th)
        # plot_within_population_clusters(X_down,dates_down, pop_ids_down, event_stage_labels, explained_variance, title=f"Clustering by Demographic Events over {th} (PCA Space)")

        number_of_subpopulations = np.unique(event_stage_labels).size
        treshold_number_of_subpopulations[th] = number_of_subpopulations

        temporal_weight_values = [0, 0.1,0.2, 0.5, 0.75, 1, 10]
        k_values = np.arange(1, 41, 1)
        scores_for_t = calculate_ARI_vs_t(X_down, temporal_weight_values, k_values, pop_ids_down, event_stage_labels,
                                          explained_variance)
        # plot_ARI_vs_t(scores_for_t, temporal_weight_values, k_values)
        threshold_scores_for_t[th] = scores_for_t

        temporal_weight_values = np.arange(0, 1.5, 0.05)
        k_values = [5, 10, 20, 30, 40, 100]
        scores_for_k = calculate_ARI_vs_k(X_down, temporal_weight_values, k_values, pop_ids_down, event_stage_labels,
                                          explained_variance)
        # plot_ARI_vs_k(scores_for_k, temporal_weight_values, k_values)
        threshold_scores_for_k[th] = scores_for_k

    # plot_kmeans_clustering(X, temporal_weight=0.1, genetic_weights=explained_variance, k=10, dates_array=dates_array)
    # plot_kmeans_clustering(X, temporal_weight=0.1, genetic_weights=explained_variance, k=25, dates_array=dates_array)

    # plot_average_score_vs_t_and_k(populations, migrations, X_down, pop_ids_down, dates_down, explained_variance)

    return threshold_scores_for_k, threshold_scores_for_t, treshold_number_of_subpopulations


def create_mean_ari_grid_heatmap(scores_for_t, temporal_weight_values, k_values):
    """
    Calculates the mean ARI score for every (K, t) combination within each threshold
    and displays them in a 3x2 grid of heatmaps (one subplot per threshold).
    """

    thresholds = sorted(scores_for_t[0].keys())
    num_thresholds = len(thresholds)  # Should be 5 based on your image

    if num_thresholds != 5:
        print("Warning: This layout is optimized for exactly 5 thresholds.")

    num_t = len(temporal_weight_values)
    num_k = len(k_values)

    # Define the grid layout: 3 rows, 2 columns (6 total slots for 5 plots)
    GRID_ROWS = 3
    GRID_COLS = 2

    # 1. Setup the 3x2 subplot grid
    # figsize adjusted to be wider than it is tall
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(18, 12), sharex=True, sharey=True)
    # Flatten the axes array for easy indexing: axes[0], axes[1], ...
    axes = axes.flatten()

    # Determine the common color scale limits (vmin, vmax)
    all_ari_values = [np.array(sim[th]).mean(axis=0) for th in thresholds for sim in scores_for_t]
    vmin = np.min(all_ari_values) if all_ari_values else 0.0
    vmax = np.max(all_ari_values) if all_ari_values else 1.0

    # 2. Iterate through each threshold and draw its heatmap
    for idx, th in enumerate(thresholds):
        ax = axes[idx]

        # Extract and average the data for the current threshold
        th_scores_for_t = [sim[th] for sim in scores_for_t]
        th_scores_for_t = np.array(th_scores_for_t)
        mean_t = th_scores_for_t.mean(axis=0)

        # Plot the heatmap (mean_t)
        im = ax.imshow(mean_t, cmap='plasma', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)

        # Set Y-axis (Temporal weight)
        ax.set_yticks(np.arange(num_t))
        ax.set_yticklabels([f'{t}' for t in temporal_weight_values])
        ax.set_ylabel('Temporal weight (t)', fontsize=10)
        ax.set_title(f'Threshold = {th}', fontsize=12)

        # Add a common Colorbar for this specific plot
        # We only add a colorbar to the first subplot in the last row for space efficiency
        if idx == len(thresholds) - 1 or idx == 0:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Mean ARI Score', rotation=-90, va="bottom")

        # Set X-axis (K values) for all plots
        ax.set_xticks(np.arange(num_k))
        # Only show labels on the bottom row for cleaner look
        if idx >= GRID_ROWS * GRID_COLS - GRID_COLS:
            ax.set_xticklabels([f'{k:.0f}' for k in k_values], rotation=45, ha="right")
            ax.set_xlabel('Number of clusters (K)', fontsize=12)
        else:
            ax.set_xticklabels([])

        # Add text annotations (optional, for high-value cells)
        for i in range(num_t):
            for j in range(num_k):
                val = mean_t[i, j]
                if val > 0.6:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="k", fontsize=5)

    # 3. Handle the empty subplot (slot 5 in a 3x2 grid for 5 plots)
    if num_thresholds < GRID_ROWS * GRID_COLS:
        # Turn off the empty subplot's axes
        axes[num_thresholds].axis('off')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    plt.suptitle("Mean ARI Score Grid: Optimal K and t per Threshold", fontsize=16)
    plt.show()