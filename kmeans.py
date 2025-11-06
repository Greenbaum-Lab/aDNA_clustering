from imports_constants_paramaters import *
from simulation import generate_kmeans_data


def downsample_within_populations(X, dates_array, population_ids_array, population_timesteps_array,
                                  max_samples_per_bin=10, bin_width=500, seed=42):
    """
    Performs stratified downsampling within each population across time bins.

    Returns:
        X_down, dates_down, pop_ids_down, timesteps_down
    """
    np.random.seed(seed)

    dates_array = np.array(dates_array)
    population_ids_array = np.array(population_ids_array)
    population_timesteps_array = np.array(population_timesteps_array)

    min_date = int(np.min(dates_array))
    max_date = int(np.max(dates_array))
    bins = np.arange(min_date, max_date + bin_width, bin_width)

    selected_indices = []

    for pop in np.unique(population_ids_array):
        pop_mask = (population_ids_array == pop)
        pop_indices = np.where(pop_mask)[0]
        pop_dates = dates_array[pop_indices]

        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i + 1]
            in_bin = (pop_dates >= bin_start) & (pop_dates < bin_end)
            bin_indices = pop_indices[in_bin]

            if len(bin_indices) > 0:
                if len(bin_indices) > max_samples_per_bin:
                    sampled = np.random.choice(bin_indices, max_samples_per_bin, replace=False)
                else:
                    sampled = bin_indices
                selected_indices.extend(sampled)

    selected_indices = np.array(sorted(selected_indices))

    return (
        X[selected_indices],
        dates_array[selected_indices],
        population_ids_array[selected_indices],
        population_timesteps_array[selected_indices]
    )


def plot_within_population_clusters(
        X,
        dates_array,
        population_ids_array,
        within_labels,
        explained_variance=None,
        title=None
):
    pc1 = X[:, 0]
    pc2 = X[:, 1]

    unique_labels = np.unique(within_labels)

    # Step 1: Extract population ID from cluster label (e.g., '0_3' -> 0)
    label_to_pop = {label: int(str(label).split('_')[0]) for label in unique_labels}
    pops = sorted(set(label_to_pop.values()))

    # Step 2: Define color palettes per population (dark to light)
    pop_to_colors = {
        0: ['#c6dbef', '#6baed6', '#08306b', '#9ecae1', '#4292c6', '#2171b5'],  # Blue
        1: ['#feedde', '#fdae6b', '#d94801', '#fdd0a2', '#fd8d3c', '#f16913'],  # Orange
        2: ['#a1d99b', '#41ab5d', '#00441b', '#74c476', '#238b45', '#1b7837'],  # Green
        3: ['#fcae91', '#ef3b2c', '#67000d', '#fb6a4a', '#cb181d', '#a50f15'],  # Red
        4: ['#bcbddc', '#807dba', '#3f007d', '#9e9ac8', '#6a51a3', '#54278f'],  # Purple
        5: ['#d4dde6', '#6c8597', '#102c3f', '#a1b3c4', '#47697f', '#2c4a5f'],  # Gray-Blue
        6: ['#fccde5', '#f768a1', '#7a0177', '#fa9fb5', '#dd3497', '#ae017e'],  # Pink
    }

    # Step 3: Calculate average date for each cluster
    cluster_avg_dates = {}
    for label in unique_labels:
        mask = (within_labels == label)
        cluster_avg_dates[label] = np.mean(dates_array[mask])

    # Step 4: Assign colors so that clusters with earlier dates get lighter colors
    label_to_color = {}
    pop_label_groups = defaultdict(list)
    for label, pop in label_to_pop.items():
        pop_label_groups[pop].append(label)

    for pop, labels_in_pop in pop_label_groups.items():
        # Sort clusters by average date descending (earlier = larger year before present)
        labels_sorted = sorted(labels_in_pop, key=lambda l: cluster_avg_dates[l], reverse=True)

        colors = pop_to_colors[pop]
        n_colors = len(colors)

        # Assign lighter colors (end of palette) to earlier clusters, darker (start) to later
        for idx, label in enumerate(labels_sorted):
            color_idx = min(idx, n_colors - 1)
            # Reverse color indexing: idx=0 gets lightest (last color), idx=n gets darkest
            label_to_color[label] = colors[-(color_idx + 1)]

    # === Plotting ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # View 1: PC1 vs PC2
    for label in unique_labels:
        mask = (within_labels == label)
        axes[0].scatter(pc1[mask], pc2[mask], color=label_to_color[label], label=label, alpha=0.9, s=30)
    axes[0].set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}%)' if explained_variance is not None else 'PC1')
    axes[0].set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}%)' if explained_variance is not None else 'PC2')
    axes[0].set_title('PC1 vs PC2')

    # View 2: PC1 vs Time
    for label in unique_labels:
        mask = (within_labels == label)
        axes[1].scatter(dates_array[mask], pc1[mask], color=label_to_color[label], alpha=0.9, s=20)
    axes[1].set_xlabel('Years Before Present')
    axes[1].set_ylabel('PC1')
    axes[1].set_title('PC1 over Time')
    axes[1].invert_xaxis()

    # View 3: PC2 vs Time
    for label in unique_labels:
        mask = (within_labels == label)
        axes[2].scatter(dates_array[mask], pc2[mask], color=label_to_color[label], alpha=0.9, s=20)
    axes[2].set_xlabel('Years Before Present')
    axes[2].set_ylabel('PC2')
    axes[2].set_title('PC2 over Time')
    axes[2].invert_xaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title='pop_cluster')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def label_by_demographic_events(population_ids_array, dates_array, migrations_data, splits, replacements, threshold=0,
                                generation_time=25):
    """
    Assigns integer stage labels (0, 1, 2, ...) to each sample based on the number and timing
    of demographic events (migration, split, replacement) that occurred in its population.

    Parameters:
    - population_ids_array : np.ndarray of population IDs for each sample
    - dates_array : np.ndarray of years before present for each sample
    - migrations, splits, replacements : lists of demographic events from the simulation
    - generation_time : int, years per generation (default: 25)

    Returns:
    - stage_labels : np.ndarray of string labels in the format "<pop_id>_<stage_number>"
    """
    labels = {}
    for pop in np.unique(population_ids_array):
        labels[pop] = [0]

    for migration in migrations_data:
        if (migration['m_i'] * migration['D_i']) >= threshold:
            pop = migration['tgt']
            year = migration['year']
            labels[pop].append(year)

        # pop=migration[SRC_POP]
        # labels[pop].append(year)

    for split in splits:
        pop = split[SRC_POP]
        year = (generations - split[SPLIT_GEN]) * generation_time
        labels[pop].append(year)

    for replacement in replacements:
        pop = replacement[TGT_POP]
        year = (generations - replacement[RPC_GEN]) * generation_time
        labels[pop].append(year)

        # pop=replacement[SRC_POP]
        # labels[pop].append(year)

    # print(f"num of clusters - {sum(len(v) for v in labels.values())}")

    dempographic_labels = np.empty(len(population_ids_array), dtype=object)
    for sample in range(len(population_ids_array)):
        pop = population_ids_array[sample]
        year = dates_array[sample]

        labels[pop].sort()

        for i in range(len(labels[pop])):
            if year > labels[pop][i]:
                dempographic_labels[sample] = f"{pop}_{i}"

    return dempographic_labels


# ----------------------------
# 2. Data Preprocessing
# ----------------------------


def run_kmeans(X, k, temporal_weight=1.0, genetic_weights=np.array([1.0, 1.0])):
    # Create a weight array matching the feature order
    # ['PC1', 'PC2', 'date']
    feature_weights = np.array(list(genetic_weights) + [temporal_weight])

    # Standardize the features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply feature weighting by scaling
    X_weighted_scaled = X_scaled * feature_weights

    # ----------------------------
    # 3. K-Means Clustering
    # ----------------------------

    # Initialize and fit the K-Means model
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        max_iter=300,
        tol=1e-4,
        random_state=42,
        n_init=1
    )
    kmeans.fit(X_weighted_scaled)

    # Assign cluster labels to each sample
    clusters = kmeans.labels_

    # Extract the cluster centroids
    centroids_scaled = kmeans.cluster_centers_

    # To interpret centroids in the original feature space, inverse transform
    with np.errstate(divide='ignore', invalid='ignore'):
        centroids_unweighted = centroids_scaled / feature_weights
    centroids = scaler.inverse_transform(centroids_unweighted)

    return clusters, centroids


def plot_kmeans_colored_by_pop(
    X,
    dates_array,
    pop_clusters,  # vector of pop_cluster labels, same length as X
    temporal_weight=1.0,
    genetic_weights=np.array([1.0, 1.0]),
    k=5
):
    clusters, centroids = run_kmeans(X, k=k, temporal_weight=temporal_weight, genetic_weights=genetic_weights)

    pc1 = X[:, 0]
    pc2 = X[:, 1]

    # Step 1: Map each cluster to its dominant population
    cluster_to_pop = {}
    for cluster_id in range(k):
        mask = clusters == cluster_id
        dominant_pop = Counter(pop_clusters[mask]).most_common(1)[0][0]
        cluster_to_pop[cluster_id] = dominant_pop

    # Step 2: Define colors for each population
    pop_to_colors = {
    0: ['#c6dbef', '#6baed6', '#08306b', '#9ecae1', '#4292c6', '#2171b5'],  # Blue
    1: ['#feedde', '#fdae6b', '#d94801', '#fdd0a2', '#fd8d3c', '#f16913'],  # Orange
    2: ['#a1d99b', '#41ab5d', '#00441b', '#74c476', '#238b45', '#1b7837'],  # Green
    3: ['#fcae91', '#ef3b2c', '#67000d', '#fb6a4a', '#cb181d', '#a50f15'],  # Red
    4: ['#bcbddc', '#807dba', '#3f007d', '#9e9ac8', '#6a51a3', '#54278f'],  # Purple
    5: ['#d4dde6', '#6c8597', '#102c3f', '#a1b3c4', '#47697f', '#2c4a5f'],  # Gray-Blue
    6: ['#fccde5', '#f768a1', '#7a0177', '#fa9fb5', '#dd3497', '#ae017e'],  # Pink
}

    # Step 3: Calculate average date for each cluster
    cluster_avg_dates = []
    for cluster_id in range(k):
        mask = clusters == cluster_id
        mean_date = np.mean(dates_array[mask])
        cluster_avg_dates.append((cluster_id, mean_date))

    # Sort clusters by average date descending (earlier times first)
    # Assuming dates_array is years before present: larger means older
    cluster_avg_dates.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Assign colors to clusters, lighter colors to earlier clusters
    cluster_colors = {}
    pop_color_usage = defaultdict(int)

    for rank, (cluster_id, _) in enumerate(cluster_avg_dates):
        dominant_pop = cluster_to_pop[cluster_id]
        colors_list = pop_to_colors[dominant_pop]
        # Assign color index so lighter colors (at end of list) correspond to earlier clusters
        color_index = len(colors_list) - 1 - (pop_color_usage[dominant_pop] % len(colors_list))
        cluster_colors[cluster_id] = colors_list[color_index]
        pop_color_usage[dominant_pop] += 1

    # Step 5: Map cluster labels to their assigned colors for plotting
    point_colors = [cluster_colors[cl] for cl in clusters]

    # ==== Plotting ====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PCA space scatter plot
    axes[0].scatter(pc1, pc2, c=point_colors, s=30)
    #axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', s=150, marker='X', label='Centroids')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title(f'KMeans Clustering with k={k}, temporal_weight={temporal_weight}')
    axes[0].legend()

    # PC1 vs Time scatter plot
    axes[1].scatter(dates_array, pc1, c=point_colors, s=20)
    axes[1].set_title('PC1 over Time')
    axes[1].set_title('PC1 over Time')
    axes[1].set_xlabel('Years Before Present')
    axes[1].set_ylabel('PC1')
    axes[1].invert_xaxis()

    # PC2 vs Time scatter plot
    axes[2].scatter(dates_array, pc2, c=point_colors, s=20)
    axes[2].set_title('PC2 over Time')
    axes[2].set_xlabel('Years Before Present')
    axes[2].set_ylabel('PC2')
    axes[2].invert_xaxis()

    plt.tight_layout()
    plt.show()
