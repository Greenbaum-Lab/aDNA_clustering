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
    imesteps_array = np.array(population_timesteps_array)

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
        title=None,
        migration_lines=None,
        current_threshold=0.0
):
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    pc1 = X[:, 0]
    pc2 = X[:, 1]

    unique_labels = np.unique(within_labels)

    # --- Initialization and Color Calculation (Unchanged) ---
    label_to_pop = {label: int(str(label).split('_')[0]) for label in unique_labels}
    pops = sorted(set(label_to_pop.values()))

    # Define colors for each population (using color palettes)
    pop_to_colors = {
        # Pop 0: Blue shades (Light to Dark)
        0: ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08306b'],

        # Pop 1: Orange shades (Light to Dark)
        1: ['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801'],

        # Pop 2: Green shades (Light to Dark)
        2: ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#1b7837', '#00441b'],

        # Pop 3: Red shades (Light to Dark) - תיקון משמעותי כאן:
        3: ['#fcae91', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],

        # Pop 4: Purple shades (Light to Dark)
        4: ['#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d'],

        # Pop 5: Gray-Blue shades (Light to Dark)
        5: ['#d4dde6', '#a1b3c4', '#6c8597', '#47697f', '#2c4a5f', '#102c3f'],

        # Pop 6: Pink shades (Light to Dark)
        6: ['#fccde5', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177'],
    }

    # Calculate average date for each cluster
    cluster_avg_dates = {}
    for label in unique_labels:
        mask = (within_labels == label)
        cluster_avg_dates[label] = np.mean(dates_array[mask])

    # Assign colors based on population and date
    label_to_color = {}
    pop_label_groups = defaultdict(list)
    for label, pop in label_to_pop.items():
        pop_label_groups[pop].append(label)

    for pop, labels_in_pop in pop_label_groups.items():
        # Sort clusters by date (older first)
        labels_sorted = sorted(labels_in_pop, key=lambda l: cluster_avg_dates[l], reverse=True)
        colors = pop_to_colors[pop]
        n_colors = len(colors)
        for idx, label in enumerate(labels_sorted):
            color_idx = min(idx, n_colors - 1)
            # Assign color: uses lighter colors for earlier (older) clusters
            label_to_color[label] = colors[-(color_idx + 1)]

    # -------------------------------------------------------------------
    # Plotting setup
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    # View 1: PC1 vs PC2
    for label in unique_labels:
        mask = (within_labels == label)
        axes[0].scatter(pc1[mask], pc2[mask], color=label_to_color[label], label=label, alpha=0.9, s=30)
    axes[0].set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}%)' if explained_variance is not None else 'PC1')
    axes[0].set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}%)' if explained_variance is not None else 'PC2')
    axes[0].set_title('PC1 vs PC2')

    # --- Migration Data Preparation & Stacking Parameters ---
    filtered_lines = []
    if migration_lines:
        filtered_lines = [
            line for line in migration_lines
            if line['score'] > current_threshold
        ]

    # Stacking parameters
    AXES_LABEL_Y_BASE = -0.3
    STACK_HEIGHT_REL = 0.12
    LABEL_TIME_WIDTH_YBP = 800
    MAX_STACK_TRACKS = 6

    # View 2: PC1 vs Time & View 3: PC2 vs Time
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    # Scatter plots of data points
    for label in unique_labels:
        mask = (within_labels == label)
        axes[1].scatter(dates_array[mask], pc1[mask], color=label_to_color[label], alpha=0.9, s=20)
        axes[2].scatter(dates_array[mask], pc2[mask], color=label_to_color[label], alpha=0.9, s=20)

    # -------------------------------------------------------------
    # 3. CALL HELPER FUNCTION TO DRAW LABELS
    # -------------------------------------------------------------
    if filtered_lines:
        # PC1 vs Time
        _draw_migration_labels(
            axes=axes[1],
            filtered_lines=filtered_lines,
            pop_to_colors=pop_to_colors,
            AXES_LABEL_Y_BASE=AXES_LABEL_Y_BASE,
            STACK_HEIGHT_REL=STACK_HEIGHT_REL,
            LABEL_TIME_WIDTH_YBP=LABEL_TIME_WIDTH_YBP,
            MAX_STACK_TRACKS=MAX_STACK_TRACKS,
            is_pc2=False
        )

        # PC2 vs Time
        _draw_migration_labels(
            axes=axes[2],
            filtered_lines=filtered_lines,
            pop_to_colors=pop_to_colors,
            AXES_LABEL_Y_BASE=AXES_LABEL_Y_BASE,
            STACK_HEIGHT_REL=STACK_HEIGHT_REL,
            LABEL_TIME_WIDTH_YBP=LABEL_TIME_WIDTH_YBP,
            MAX_STACK_TRACKS=MAX_STACK_TRACKS,
            is_pc2=True
        )

    # Finalizing axes settings
    axes[1].set_xlabel('Years Before Present')
    axes[1].set_ylabel('PC1')
    axes[1].set_title('PC1 over Time')
    axes[1].invert_xaxis()

    axes[2].set_xlabel('Years Before Present')
    axes[2].set_ylabel('PC2')
    axes[2].set_title('PC2 over Time')
    axes[2].invert_xaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title='pop_cluster')

    if title:
        fig.suptitle(title + f"number of clusters - {unique_labels.shape[0]}", fontsize=14)

    # Adjust subplot parameters to make enough room for the labels (bottom set high due to -0.3 base)
    plt.subplots_adjust(bottom=0.35)
    plt.show()

def plot_demographic_events_by_subcluster(
        dates_array,
        within_labels,
        title=None,
        migration_lines=None,  # <<< שינוי: מקבל רשימה גולמית
        current_threshold=0.0  # <<< שינוי: סף הסינון
):
    """
        Plots the demographic-temporal structure found by the clustering algorithm,
        and optionally adds migration bars, filtered by a genetic threshold.

        Args:
            dates_array (np.ndarray): Array of sample dates (Years Before Present).
            within_labels (np.ndarray): Array of cluster labels (e.g., '0_1', '1_3').
            title (str, optional): Title for the plot.
            migration_lines (list of dicts, optional): Data defining ALL migration events (unfiltered).
            current_threshold (float): Only plot migrations where score > current_threshold.
        """

    # 1. Basic definitions and mapping
    unique_labels = np.unique(within_labels)
    if unique_labels.size == 0:
        print("Error: No unique labels found. Cannot plot.")
        return

    label_to_pop = {label: int(str(label).split('_')[0]) for label in unique_labels}
    pops = sorted(set(label_to_pop.values()))
    pop_to_y_pos = {pop: idx + 1 for idx, pop in enumerate(pops)}
    rect_height = 0.6

    # 2. Color Palettes: Matches the user's color mapping logic (lightest to darkest)
    pop_to_colors = {
        # Pop 0: Blue shades (Light to Dark)
        0: ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08306b'],

        # Pop 1: Orange shades (Light to Dark)
        1: ['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801'],

        # Pop 2: Green shades (Light to Dark)
        2: ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#1b7837', '#00441b'],

        # Pop 3: Red shades (Light to Dark) - תיקון משמעותי כאן:
        3: ['#fcae91', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],

        # Pop 4: Purple shades (Light to Dark)
        4: ['#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d'],

        # Pop 5: Gray-Blue shades (Light to Dark)
        5: ['#d4dde6', '#a1b3c4', '#6c8597', '#47697f', '#2c4a5f', '#102c3f'],

        # Pop 6: Pink shades (Light to Dark)
        6: ['#fccde5', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177'],
    }

    # 3. Calculate average date for each sub-cluster
    cluster_avg_dates = {}
    for label in unique_labels:
        mask = (within_labels == label)
        if np.sum(mask) > 0:
            cluster_avg_dates[label] = np.mean(dates_array[mask])
        else:
            cluster_avg_dates[label] = np.inf

    # 4. Assign specific color based on temporal ordering (CRITICAL STEP)
    label_to_color = {}
    pop_label_groups = defaultdict(list)
    for label, pop in label_to_pop.items():
        pop_label_groups[pop].append(label)

    for pop, labels_in_pop in pop_label_groups.items():
        # Sort sub-clusters by average date descending (earlier = larger YBP)
        labels_sorted = sorted(labels_in_pop, key=lambda l: cluster_avg_dates.get(l, np.inf), reverse=True)

        colors = pop_to_colors.get(pop, pop_to_colors.get(0, ['gray']))
        n_colors = len(colors)

        for idx, label in enumerate(labels_sorted):
            color_idx = min(idx, n_colors - 1)
            # Assign color: This line matches your scatter plot logic
            label_to_color[label] = colors[-(color_idx + 1)]

    # We need pop_to_representative_color for the migration bars, but we assume it's calculated in run_clustering_evaluation
    # and passed via migration_lines. However, if run_clustering_evaluation doesn't calculate it, we re-calculate it here:
    pop_to_representative_color = {pop: pop_to_colors.get(pop, ['gray'])[-1] for pop in pops if pop in pop_to_colors}
    for pop in pops:
        if pop not in pop_to_representative_color:
            pop_to_representative_color[pop] = 'gray'

    # -----------------------------------------------------
    # Plotting setup
    # -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Draw Rectangles (Bounding Boxes) ---
    for label in unique_labels:
        try:
            color = label_to_color.get(label, 'gray')
            pop = label_to_pop[label]

            y_pos_center = pop_to_y_pos[pop]
            y_start = y_pos_center - rect_height / 2

            mask = (within_labels == label)

            if np.sum(mask) == 0:
                continue

            min_date = np.min(dates_array[mask])
            max_date = np.max(dates_array[mask])

            x_start = max_date
            width = min_date - max_date
            height = rect_height

            rect = patches.Rectangle(
                (x_start, y_start),
                width,
                height,
                linewidth=1.5,
                edgecolor='black',
                facecolor=color,
                alpha=0.7,
                zorder=2
            )
            ax.add_patch(rect)

        except Exception as e:
            if "nan" not in str(e).lower() and "inf" not in str(e).lower():
                print(f"Warning: Failed to plot label {label} due to data error: {e}")
            continue

    # --- Draw Migration Vertical Bars (with local filtering) ---
    if migration_lines:

        # 1. Filter lines based on the plot's threshold
        filtered_lines = [
            line for line in migration_lines
            if line['score'] > current_threshold
        ]

        print(
            f"Plotting {len(filtered_lines)} migration events (Score > {current_threshold:.5f}) on ground truth plot.")

        for line_data in filtered_lines:

            y_pos_center = line_data['y_pos_pop_plot']

            if y_pos_center is not None:
                # rect_height is defined above
                y_start = y_pos_center - rect_height / 2
                y_end = y_pos_center + rect_height / 2
                mig_year = line_data['year']

                # 1. שרטוט הקו האנכי (הפס)
                ax.plot(
                    [mig_year, mig_year],
                    [y_start, y_end],
                    color=line_data['color'],
                    linestyle='-',
                    linewidth=2.5,
                    zorder=4,
                    alpha=1.0
                )

                # 2. הוספת התווית
                ax.text(
                    mig_year,
                    y_end + 0.05,
                    line_data['score_text'],
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    color=line_data['color'],
                    fontweight='bold',
                    zorder=5
                )

    # -----------------------------------------------------
    # Final Aesthetic Settings and Autoscale Fix
    # -----------------------------------------------------

    ax.autoscale_view()

    ax.set_xlabel('Years Before Present (YBP)')
    ax.invert_xaxis()
    ax.set_ylabel('Population ID (Clustered)')

    y_ticks = [pop_to_y_pos[pop] for pop in pops]
    y_labels = [f'Pop {pop + 1}' for pop in pops]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)

    for y in np.array(y_ticks) + 0.5:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, zorder=1)

    if title:
        ax.set_title(title)

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


def plot_ellipse(ax, mean, cov, color, alpha=0.7, n_std=1.5):
    """
    Draws an ellipse representing a multivariate normal distribution at an n_std confidence level.
    The ellipse is based on the mean and covariance matrix of the data cluster.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the ellipse on.
        mean (numpy.ndarray): The mean vector (center) of the data.
        cov (numpy.ndarray): The covariance matrix of the data.
        color (str): The color of the ellipse.
        alpha (float): The transparency level for the fill.
        n_std (float): The number of standard deviations for the ellipse size.
    """
    # Calculate eigenvalues and eigenvectors of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # Calculate the angle of the main axis (angle of the larger eigenvector)
    # The larger eigenvector determines the angle of the ellipse's major axis
    angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))

    # Calculate the length of the axes (square root of the eigenvalues, multiplied by n_std)
    width, height = 2 * n_std * np.sqrt(eig_vals)

    # Create the ellipse
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        color=color,
        alpha=alpha,
        fill=True,
        linewidth=0,  # No outline
        zorder=2  # UPDATED: Ensure the ellipse is IN FRONT of the scattered points
    )
    ax.add_patch(ellipse)
    return ellipse



# Assuming run_kmeans, plot_ellipse, and other necessary libraries (KMeans, StandardScaler) are imported elsewhere.


def plot_kmeans_colored_by_pop(
        X,
        dates_array,
        pop_clusters,  # vector of pop_cluster labels, same length as X
        temporal_weight=1.0,
        genetic_weights=np.array([1.0, 1.0]),
        k=5,
        migration_lines=None  # NEW: Argument for migration data
):
    """
    Performs K-means clustering ONCE, plots the PCA results, and then calls
    plot_kmeans_demographic_style to display the second visualization.
    Includes ALL migration labels (threshold 0) if migration_lines is provided.
    """


    # 1. === Run K-means Clustering ONCE ===

    # Determine the number of genetic features to use based on the length of genetic_weights.
    num_genetic_features = len(genetic_weights)
    X_sliced = X[:, :num_genetic_features]  # Slice X (PCA) to match the number of genetic weights

    # Combine the SLICED PCA data and time data.
    X_for_kmeans = np.column_stack([X_sliced, dates_array])

    # Run K-means with the matched data
    # NOTE: Assuming run_kmeans and the definitions of clusters/centroids are correct and accessible.
    clusters, centroids = run_kmeans(X_for_kmeans, k=k, temporal_weight=temporal_weight,
                                     genetic_weights=genetic_weights)
    # ======================================

    pc1 = X[:, 0]
    pc2 = X[:, 1]

    # Step 1: Map each cluster to its dominant population (for consistent coloring)
    cluster_to_pop = {}
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            dominant_pop = Counter(pop_clusters[mask]).most_common(1)[0][0]
            cluster_to_pop[cluster_id] = dominant_pop
        else:
            cluster_to_pop[cluster_id] = -1

    # Step 2: Define colors for each population (using color palettes)
    pop_to_colors = {
        # Pop 0: Blue shades (Light to Dark)
        0: ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08306b'],

        # Pop 1: Orange shades (Light to Dark)
        1: ['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801'],

        # Pop 2: Green shades (Light to Dark)
        2: ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#1b7837', '#00441b'],

        # Pop 3: Red shades (Light to Dark) - תיקון משמעותי כאן:
        3: ['#fcae91', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],

        # Pop 4: Purple shades (Light to Dark)
        4: ['#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d'],

        # Pop 5: Gray-Blue shades (Light to Dark)
        5: ['#d4dde6', '#a1b3c4', '#6c8597', '#47697f', '#2c4a5f', '#102c3f'],

        # Pop 6: Pink shades (Light to Dark)
        6: ['#fccde5', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177'],
    }

    # Step 3: Calculate average date for each cluster and sort by date
    cluster_avg_dates = []
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            mean_date = np.mean(dates_array[mask])
            cluster_avg_dates.append((cluster_id, mean_date))
    cluster_avg_dates.sort(key=lambda x: x[1], reverse=True)  # Sort by date (older = larger YBP = first in list)

    # Step 4: Assign colors to clusters, lighter colors to earlier clusters
    cluster_colors = {}
    pop_color_usage = defaultdict(int)
    for rank, (cluster_id, _) in enumerate(cluster_avg_dates):
        dominant_pop = cluster_to_pop[cluster_id]
        if dominant_pop != -1:
            colors_list = pop_to_colors[dominant_pop]
            color_index = pop_color_usage[dominant_pop] % len(colors_list)
            # Assign color: uses lighter colors (earlier in list) for earlier clusters
            cluster_colors[cluster_id] = colors_list[color_index]
            pop_color_usage[dominant_pop] += 1
        else:
            cluster_colors[cluster_id] = '#808080'
    point_colors = [cluster_colors[cl] for cl in clusters]

    # ==== Plotting PCA Graphs (The original 3 plots) ====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loop over all clusters to draw points and ellipses (Logic remains the same)
    for cluster_id in range(k):
        mask = clusters == cluster_id
        cluster_color = cluster_colors[cluster_id]
        if np.sum(mask) <= 1:
            if np.sum(mask) == 1:
                axes[0].scatter(pc1[mask], pc2[mask], c=cluster_color, s=30, zorder=1)
                axes[1].scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20, zorder=1)
                axes[2].scatter(dates_array[mask], pc2[mask], c=cluster_color, s=20, zorder=1)
            continue

        # Drawing on axes[0] - PC1 vs PC2 (PCA space)
        ax = axes[0]
        cluster_data_pca = np.stack((pc1[mask], pc2[mask]), axis=-1);
        mean_pca = np.mean(cluster_data_pca, axis=0);
        cov_pca = np.cov(cluster_data_pca, rowvar=False)
        plot_ellipse(ax, mean_pca, cov_pca, cluster_color, alpha=0.7, n_std=1.5);
        ax.scatter(pc1[mask], pc2[mask], c=cluster_color, s=30, zorder=1)

        # Drawing on axes[1] - PC1 vs Time
        ax = axes[1]
        cluster_data_time_pc1 = np.stack((dates_array[mask], pc1[mask]), axis=-1);
        mean_time_pc1 = np.mean(cluster_data_time_pc1, axis=0);
        cov_time_pc1 = np.cov(cluster_data_time_pc1, rowvar=False)
        plot_ellipse(ax, mean_time_pc1, cov_time_pc1, cluster_color, alpha=0.7, n_std=1.5);
        ax.scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20, zorder=1)

        # Drawing on axes[2] - PC2 vs Time
        ax = axes[2]
        cluster_data_time_pc2 = np.stack((dates_array[mask], pc2[mask]), axis=-1);
        mean_time_pc2 = np.mean(cluster_data_time_pc2, axis=0);
        cov_time_pc2 = np.cov(cluster_data_time_pc2, rowvar=False)
        plot_ellipse(ax, mean_time_pc2, cov_time_pc2, cluster_color, alpha=0.7, n_std=1.5);
        ax.scatter(dates_array[mask], pc2[mask], c=cluster_color, s=20, zorder=1)

    # -------------------------------------------------------------
    # NEW: Add Migration Labels (Ground Truth)
    # -------------------------------------------------------------
    if migration_lines is not None:
        # Define stacking parameters (these must match the logic in _draw_migration_labels)
        AXES_LABEL_Y_BASE = -0.3
        STACK_HEIGHT_REL = 0.12
        LABEL_TIME_WIDTH_YBP = 800
        MAX_STACK_TRACKS = 6

        # Filter is not necessary since we want score > 0 (all lines)
        filtered_lines = migration_lines

        # 1. PC1 vs Time
        _draw_migration_labels(
            axes=axes[1],
            filtered_lines=filtered_lines,
            pop_to_colors=pop_to_colors,
            AXES_LABEL_Y_BASE=AXES_LABEL_Y_BASE,
            STACK_HEIGHT_REL=STACK_HEIGHT_REL,
            LABEL_TIME_WIDTH_YBP=LABEL_TIME_WIDTH_YBP,
            MAX_STACK_TRACKS=MAX_STACK_TRACKS,
            is_pc2=False
        )

        # 2. PC2 vs Time
        _draw_migration_labels(
            axes=axes[2],
            filtered_lines=filtered_lines,
            pop_to_colors=pop_to_colors,
            AXES_LABEL_Y_BASE=AXES_LABEL_Y_BASE,
            STACK_HEIGHT_REL=STACK_HEIGHT_REL,
            LABEL_TIME_WIDTH_YBP=LABEL_TIME_WIDTH_YBP,
            MAX_STACK_TRACKS=MAX_STACK_TRACKS,
            is_pc2=True
        )

        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[2].grid(True, linestyle='--', alpha=0.6)

    # Finalizing axes settings and titles
    axes[0].set_xlabel('PC1');
    axes[0].set_ylabel('PC2');
    axes[0].set_title(f'KMeans Clustering in PCA space k={k}, temporal_weight={temporal_weight}')
    axes[1].set_title('PC1 over Time');
    axes[1].set_xlabel('Years Before Present');
    axes[1].set_ylabel('PC1');
    axes[1].invert_xaxis()
    axes[2].set_title('PC2 over Time');
    axes[2].set_xlabel('Years Before Present');
    axes[2].set_ylabel('PC2');
    axes[2].invert_xaxis()

    plt.tight_layout()
    # Adjust subplot bottom only if labels were actually drawn
    if migration_lines is not None:
        plt.subplots_adjust(bottom=0.35)
    plt.show()


def prepare_migration_lines(
        migration_events_data,
        pop_to_mean_pc1,
        pop_to_mean_pc2,
        pop_to_y_pos,
        pop_to_representative_color
):
    """
    Calculates and structures ALL migration events data for plotting,
    including the score (m_i * D_i), without applying any threshold filter.
    Returns a list of dicts.
    """

    if migration_events_data is None:
        return []

    migration_lines = []

    for event in migration_events_data:
        src_pop = event['src']
        tgt_pop = event['tgt']
        mig_year = event['year']
        # Calculate the score for later filtering
        score = event.get('m_i', 0) * event.get('D_i', 0)

        line_data = {
            'year': mig_year,
            'score': score,
            'score_text': f'{score:.3f}',
            'color': pop_to_representative_color.get(src_pop, 'black'),
            'src_pop': src_pop,
            'tgt_pop': tgt_pop,

            # Y position for the three plots:
            'y_pos_pop_plot': pop_to_y_pos.get(tgt_pop),
            'y_pos_pc1_plot': pop_to_mean_pc1.get(tgt_pop),
            'y_pos_pc2_plot': pop_to_mean_pc2.get(tgt_pop),
        }
        migration_lines.append(line_data)

    return migration_lines


# This function handles the logic for drawing migration labels with waterfall stacking.
# It should be placed before the main plotting functions.
def _draw_migration_labels(
        axes,
        filtered_lines,
        pop_to_colors,
        AXES_LABEL_Y_BASE,
        STACK_HEIGHT_REL,
        LABEL_TIME_WIDTH_YBP,
        MAX_STACK_TRACKS,
        is_pc2=False  # True for PC2 (uses y_pos_pc2_plot), False for PC1
):
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Waterfall Stacking Logic ---
    def get_waterfall_offset(mig_year, occupied_tracks):
        """Calculates the vertical offset (track level) for a label."""
        for track_level in range(MAX_STACK_TRACKS):
            if track_level >= len(occupied_tracks):
                occupied_tracks.append(mig_year + LABEL_TIME_WIDTH_YBP)
                return track_level

            if mig_year > occupied_tracks[track_level]:
                occupied_tracks[track_level] = mig_year + LABEL_TIME_WIDTH_YBP
                return track_level
        return MAX_STACK_TRACKS

    # --- Drawing Logic ---
    sorted_lines = sorted(filtered_lines, key=lambda x: x['year'], reverse=False)
    y_lim = axes.get_ylim()
    # Get the transform object from the axes
    trans = axes.get_xaxis_transform()
    occupied_tracks = []

    for line_data in sorted_lines:
        # Determine the correct Y position key
        y_pos_key = 'y_pos_pc2_plot' if is_pc2 else 'y_pos_pc1_plot'

        # Proceed only if Y position data exists for this graph (PC1 or PC2)
        if line_data.get(y_pos_key) is not None:
            mig_year = line_data['year']

            # Calculate Target Population Color (assuming key 0-6 is used for map lookup)
            target_pop_id = line_data['tgt_pop']
            # Use the second color in the list as the representative shade for the target pop
            target_pop_color = pop_to_colors.get(target_pop_id, pop_to_colors[0])[1]

            # Calculate offset and the relative position of the label
            y_offset = get_waterfall_offset(mig_year, occupied_tracks)
            label_y_rel = AXES_LABEL_Y_BASE - y_offset * STACK_HEIGHT_REL

            # Format the label text (including adding +1 for 1-7 display)
            src_pop_index = [k for k, v in pop_to_colors.items() if line_data['color'] in v]
            src_pop_index = src_pop_index[0] if src_pop_index else 'Src'
            label_text = f"{src_pop_index + 1} -> {target_pop_id + 1}\n{line_data['score_text']}"

            # 1. Draw the full vertical line
            axes.plot(
                [mig_year, mig_year],
                y_lim,
                color=target_pop_color,
                linestyle='--',
                linewidth=0.5,
                zorder=3,
                alpha=0.7
            )

            # 2. Draw the label
            axes.text(
                mig_year, label_y_rel, label_text,
                fontsize=7, ha='center', va='bottom',
                color=target_pop_color,
                fontweight='bold', zorder=6, transform=trans,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle="round,pad=0.3")
            )
    # Restore Y-limits after drawing the lines
    axes.set_ylim(y_lim)


def plot_kmeans_pc1_vs_time_only(
        X,
        dates_array,
        pop_clusters,
        temporal_weight=1.0,
        genetic_weights=np.array([1.0, 1.0]),
        k=5,
        migration_lines=None
):
    # 1. === Run K-means Clustering ONCE ===

    # Determine the number of genetic features to use based on the length of genetic_weights.
    num_genetic_features = len(genetic_weights)
    X_sliced = X[:, :num_genetic_features]  # Slice X (PCA) to match the number of genetic weights

    # Combine the SLICED PCA data and time data.
    X_for_kmeans = np.column_stack([X_sliced, dates_array])

    # Run K-means with the matched data
    # NOTE: Assuming run_kmeans and the definitions of clusters/centroids are correct and accessible.
    clusters, centroids = run_kmeans(X_for_kmeans, k=k, temporal_weight=temporal_weight,
                                     genetic_weights=genetic_weights)
    # ======================================

    pc1 = X[:, 0]
    pc2 = X[:, 1]

    # Step 1: Map each cluster to its dominant population (for consistent coloring)
    cluster_to_pop = {}
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            dominant_pop = Counter(pop_clusters[mask]).most_common(1)[0][0]
            cluster_to_pop[cluster_id] = dominant_pop
        else:
            cluster_to_pop[cluster_id] = -1

    # Step 2: Define colors for each population (using color palettes)
    pop_to_colors = {
        # Pop 0: Blue shades (Light to Dark)
        0: ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08306b'],

        # Pop 1: Orange shades (Light to Dark)
        1: ['#feedde', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801'],

        # Pop 2: Green shades (Light to Dark)
        2: ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#1b7837', '#00441b'],

        # Pop 3: Red shades (Light to Dark) - תיקון משמעותי כאן:
        3: ['#fcae91', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],

        # Pop 4: Purple shades (Light to Dark)
        4: ['#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d'],

        # Pop 5: Gray-Blue shades (Light to Dark)
        5: ['#d4dde6', '#a1b3c4', '#6c8597', '#47697f', '#2c4a5f', '#102c3f'],

        # Pop 6: Pink shades (Light to Dark)
        6: ['#fccde5', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177'],
    }

    # Step 3: Calculate average date for each cluster and sort by date
    cluster_avg_dates = []
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            mean_date = np.mean(dates_array[mask])
            cluster_avg_dates.append((cluster_id, mean_date))
    cluster_avg_dates.sort(key=lambda x: x[1], reverse=True)  # Sort by date (older = larger YBP = first in list)

    # Step 4: Assign colors to clusters, lighter colors to earlier clusters
    cluster_colors = {}
    pop_color_usage = defaultdict(int)
    for rank, (cluster_id, _) in enumerate(cluster_avg_dates):
        dominant_pop = cluster_to_pop[cluster_id]
        if dominant_pop != -1:
            colors_list = pop_to_colors[dominant_pop]
            color_index = pop_color_usage[dominant_pop] % len(colors_list)
            # Assign color: uses lighter colors (earlier in list) for earlier clusters
            cluster_colors[cluster_id] = colors_list[color_index]
            pop_color_usage[dominant_pop] += 1
        else:
            cluster_colors[cluster_id] = '#808080'
    point_colors = [cluster_colors[cl] for cl in clusters]

    # ==== Plotting PCA Graphs (Only PC1 vs Time) ====
    fig, ax = plt.subplots(figsize=(9, 5))  # Only one plot for PC1 vs Time

    # Loop over all clusters to draw points and ellipses (Logic remains the same)
    for cluster_id in range(k):
        mask = clusters == cluster_id
        cluster_color = cluster_colors[cluster_id]
        if np.sum(mask) <= 1:
            if np.sum(mask) == 1:
                ax.scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20, zorder=1)
            continue

        # Drawing on the PC1 vs Time plot
        cluster_data_time_pc1 = np.stack((dates_array[mask], pc1[mask]), axis=-1)
        mean_time_pc1 = np.mean(cluster_data_time_pc1, axis=0)
        cov_time_pc1 = np.cov(cluster_data_time_pc1, rowvar=False)
        plot_ellipse(ax, mean_time_pc1, cov_time_pc1, cluster_color, alpha=0.7, n_std=1.5)
        ax.scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20, zorder=1)

    # -------------------------------------------------------------
    # NEW: Add Migration Labels (Ground Truth)
    # -------------------------------------------------------------
    if migration_lines is not None:
        # Define stacking parameters (these must match the logic in _draw_migration_labels)
        AXES_LABEL_Y_BASE = -0.3
        STACK_HEIGHT_REL = 0.12
        LABEL_TIME_WIDTH_YBP = 800
        MAX_STACK_TRACKS = 6

        # Filter is not necessary since we want score > 0 (all lines)
        filtered_lines = migration_lines

        # 1. PC1 vs Time
        _draw_migration_labels(
            axes=ax,
            filtered_lines=filtered_lines,
            pop_to_colors=pop_to_colors,
            AXES_LABEL_Y_BASE=AXES_LABEL_Y_BASE,
            STACK_HEIGHT_REL=STACK_HEIGHT_REL,
            LABEL_TIME_WIDTH_YBP=LABEL_TIME_WIDTH_YBP,
            MAX_STACK_TRACKS=MAX_STACK_TRACKS,
            is_pc2=False
        )

        ax.grid(True, linestyle='--', alpha=0.6)

    # Finalizing axes settings and titles
    ax.set_xlabel('Years Before Present')
    ax.set_ylabel('PC1')
    ax.set_title(f'KMeans Clustering (k={k}, temporal_weight={temporal_weight})')
    ax.invert_xaxis()

    plt.tight_layout()
    # Adjust subplot bottom only if labels were actually drawn
    if migration_lines is not None:
        plt.subplots_adjust(bottom=0.35)
    plt.show()