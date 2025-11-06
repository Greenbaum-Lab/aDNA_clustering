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


def evaluate_migrations(populations, migrations, generation_time=25):
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

    thresholds = [0, 0.15, 0.25, 0.4]
    # plot_kmeans_colored_by_pop(X_down, dates_array=dates_down, pop_clusters=pop_ids_down, temporal_weight=0.5, genetic_weights=explained_variance, k=25)

    threshold_scores_for_k = {}
    threshold_scores_for_t = {}


    for th in thresholds:
        print(f"############# running evaluation for threshold - {th} #############")
        event_stage_labels = label_by_demographic_events(pop_ids_down, dates_down, migrations_data, splits,
                                                         replacements, threshold=th)
        # plot_within_population_clusters(X_down,dates_down, pop_ids_down, event_stage_labels, explained_variance, title=f"Clustering by Demographic Events over {th} (PCA Space)")

        temporal_weight_values = [0, 0.1, 0.5, 0.75, 1, 10]
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

    return threshold_scores_for_k, threshold_scores_for_t

