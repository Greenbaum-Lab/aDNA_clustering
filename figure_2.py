from imports_constants_paramaters import *
from kmeans import *
from clustering_evaluation import *
from simulation import *
from main import *




def create_figure_2():
    print('Creating figure 2...')
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 12), constrained_layout=False)

    print("creating single simulation data")
    dates_down, event_stage_labels, migrations_data, splits, replacements, X_down, pop_ids_down, explained_variance, migration_lines = create_single_simulation_data()
    print(event_stage_labels)
    print("first row")
    # first row - simulation example
    # plot a: ground truth
    th = 0
    plot_ground_truth(axes[0, 0], dates_down, event_stage_labels, title=f"Ground Truth for Threshold {th}",
                                              migration_lines=migration_lines, current_threshold=th)

    #b: kmeans results - k=20 t = 0
    plot_kmeans(axes[0, 1], X_down, dates_array=dates_down, pop_clusters=pop_ids_down, temporal_weight=0,
                               genetic_weights=explained_variance, k=20,
                               migration_lines=migration_lines)

    #c: kmeans results - k=20 t = 0.75
    plot_kmeans(axes[0, 2], X_down, dates_array=dates_down, pop_clusters=pop_ids_down, temporal_weight=1,
                               genetic_weights=explained_variance, k=20,
                               migration_lines=migration_lines)

    print("second row")
    ARI_data = generate_ari_scores(num_of_simulations=10, num_workers=50)
    # second row - ARI of many simulations
    # d: ARI vs k threshold = 0
    plot_ARI(axes[1,0], threshold=0, ARI_data=ARI_data)
    # e: ARI vs k threshold = 0.004
    plot_ARI(axes[1,1], threshold=0.004, ARI_data=ARI_data)

    # f: ARI vs k threshold = 0.008
    plot_ARI(axes[1, 2], threshold=0.008, ARI_data=ARI_data)

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.95,
        bottom=0.1,  # Keep overall bottom margin reasonable
        hspace=0.6,  # Key setting: Increase vertical spacing between rows
        wspace=0.1
    )
    plt.show()

def create_single_simulation_data():
    migrations = []
    splits = []
    replacements = []
    gen = 0
    for i in range(25):
        src_pop = np.random.randint(0, 7)
        tgt_pop = np.random.randint(0, 7)
        while tgt_pop == src_pop:
            tgt_pop = np.random.randint(0, 7)
        size = np.random.uniform(0.01, 0.2)
        duration = 1
        migrations.append([src_pop, tgt_pop, gen, size, duration])
        gen += 16
    # Function to create data for a single simulation
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

    # --- Calculation of necessary data for migration line plotting ---
    pops = sorted(np.unique(pop_ids_down))
    pop_to_y_pos = {pop: idx + 1 for idx, pop in enumerate(pops)}
    pop_to_colors = {
        0: ['#c6dbef', '#6baed6', '#08306b', '#d94801', '#238b45', '#a50f15', '#54278f'],
        1: ['#feedde', '#fdae6b', '#d94801', '#fdd0a2', '#fd8d3c', '#f16913'],
        2: ['#a1d99b', '#41ab5d', '#00441b', '#74c476', '#238b45', '#1b7837'],
        3: ['#fcae91', '#ef3b2c', '#67000d', '#fb6a4a', '#cb181d', '#a50f15'],
        4: ['#bcbddc', '#807dba', '#3f007d', '#9e9ac8', '#6a51a3', '#54278f'],
        5: ['#d4dde6', '#6c8597', '#102c3f', '#a1b3c4', '#47697f', '#2c4a5f'],
        6: ['#fccde5', '#f768a1', '#7a0177', '#fa9fb5', '#dd3497', '#ae017e'],
    }  # Note: The full list of colors is not strictly needed here, only the list of populations and colors for migration lines.

    pop_to_representative_color = {pop: pop_to_colors.get(pop, ['gray'])[-1] for pop in pops if
                                   pop in pop_to_colors}
    for pop in pops:
        if pop not in pop_to_representative_color:
            pop_to_representative_color[pop] = 'gray'

    pc1 = X_down[:, 0]
    pc2 = X_down[:, 1]
    pop_to_mean_pc1 = {}
    pop_to_mean_pc2 = {}

    for pop in pops:
        mask = (pop_ids_down == pop)
        if np.sum(mask) > 0:
            pop_to_mean_pc1[pop] = np.mean(pc1[mask])
            pop_to_mean_pc2[pop] = np.mean(pc2[mask])
        else:
            pop_to_mean_pc1[pop] = 0
            pop_to_mean_pc2[pop] = 0

    migration_lines = prepare_migration_lines(
        migrations_data,
        pop_to_mean_pc1,
        pop_to_mean_pc2,
        pop_to_y_pos,
        pop_to_representative_color
    )
    event_stage_labels = label_by_demographic_events(pop_ids_down, dates_down, migrations_data, splits,
                                                     replacements, threshold=0)

    return (dates_down, event_stage_labels, migrations_data, splits, replacements,
            X_down, pop_ids_down, explained_variance, migration_lines)

def plot_ground_truth(ax, dates_down, event_stage_labels, title, migration_lines, current_threshold):
    """
    Plots the ground truth structure as colored rectangles over time,
    including vertical migration bars based on the current threshold.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the plot on.
        dates_down (np.ndarray): Array of dates (YBP) for the downsampled samples.
        event_stage_labels (np.ndarray): Labels assigning each sample to a demographic stage/sub-cluster.
        title (str): Title of the subplot.
        migration_lines (list): List of dictionaries containing migration line data (year, score, color, etc.).
        current_threshold (float): The score threshold for plotting migration events.
    """

    # 1. Basic definitions and mapping
    unique_labels = np.unique(event_stage_labels)
    # Filter out non-string/non-meaningful labels if they are objects
    if unique_labels.size == 0:
        print("Error: No unique labels found. Cannot plot.")
        return

    label_to_pop = {label: int(str(label).split('_')[0]) for label in unique_labels}
    pops = sorted(set(label_to_pop.values()))
    pop_to_y_pos = {pop: idx + 1 for idx, pop in enumerate(pops)}
    rect_height = 0.6

    # 2. Color Palettes: Matches the user's color mapping logic (lightest to darkest)
    pop_to_colors = {
        0: ['#c6dbef', '#6baed6', '#08306b', '#9ecae1', '#4292c6', '#2171b5'],  # Blue
        1: ['#feedde', '#fdae6b', '#d94801', '#fdd0a2', '#fd8d3c', '#f16913'],  # Orange
        2: ['#a1d99b', '#41ab5d', '#00441b', '#74c476', '#238b45', '#1b7837'],  # Green
        3: ['#fcae91', '#ef3b2c', '#67000d', '#fb6a4a', '#cb181d', '#a50f15'],  # Red
        4: ['#bcbddc', '#807dba', '#3f007d', '#9e9ac8', '#6a51a3', '#54278f'],  # Purple
        5: ['#d4dde6', '#6c8597', '#102c3f', '#a1b3c4', '#47697f', '#2c4a5f'],  # Gray-Blue
        6: ['#fccde5', '#f768a1', '#7a0177', '#fa9fb5', '#dd3497', '#ae017e'],  # Pink
    }

    # 3. Calculate average date for each sub-cluster
    cluster_avg_dates = {}
    for label in unique_labels:
        mask = (event_stage_labels == label)
        if np.sum(mask) > 0:
            cluster_avg_dates[label] = np.mean(dates_down[mask])
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
    # Plotting Rectangles (Bounding Boxes)
    # -----------------------------------------------------

    for label in unique_labels:
        if label not in label_to_pop: continue

        try:
            color = label_to_color.get(label, 'gray')
            pop = label_to_pop[label]

            y_pos_center = pop_to_y_pos[pop]
            y_start = y_pos_center - rect_height / 2

            mask = (event_stage_labels == label)

            if np.sum(mask) == 0:
                continue

            min_date = np.min(dates_down[mask])
            max_date = np.max(dates_down[mask])

            x_start = max_date
            width = min_date - max_date


            rect = patches.Rectangle(
                (x_start, y_start),
                width,
                rect_height,
                linewidth=1.5,
                edgecolor='black',
                facecolor=color,
                alpha=0.7,
                zorder=2
            )
            ax.add_patch(rect)

        except Exception as e:
            print(f"Warning: Failed to plot label {label} due to data error: {e}")
            continue

    # -----------------------------------------------------
    # Plotting Migration Vertical Bars (and Arrows - arrows logic omitted here)
    # -----------------------------------------------------
    if migration_lines:

        filtered_lines = [
            line for line in migration_lines
            if line.get('score', 0) > current_threshold
        ]

        for line_data in filtered_lines:
            y_pos_center = line_data.get('y_pos_pop_plot')

            if y_pos_center is not None:
                y_start = y_pos_center - rect_height / 2
                y_end = y_pos_center + rect_height / 2
                mig_year = line_data['year']

                # 1. Draw the vertical bar
                ax.plot(
                    [mig_year, mig_year],
                    [y_start, y_end],
                    color=line_data['color'],
                    linestyle='-',
                    linewidth=2.5,
                    zorder=4,
                    alpha=1.0
                )

                # 2. Add the score label
                score_text = line_data.get('score_text', f"{line_data.get('score', 0):.3f}")
                ax.text(
                    mig_year,
                    y_end + 0.05,
                    score_text,
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    color=line_data['color'],
                    fontweight='bold',
                    zorder=5
                )

    # -----------------------------------------------------
    # Final Aesthetic Settings
    # -----------------------------------------------------

    ax.set_xlabel('Years Before Present (YBP)')
    ax.invert_xaxis()
    ax.set_ylabel('Population ID (Clustered)')

    if pops:
        y_ticks = [pop_to_y_pos[pop] for pop in pops]
        y_labels = [f'Pop {pop + 1}' for pop in pops]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(min(y_ticks) - 0.7, max(y_ticks) + 0.7)

        # Draw horizontal grid lines between populations
        for y in np.array(y_ticks) + 0.5:
            ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5, zorder=1)

    if title:
        # Use only the title without the A. prefix here, let the caller (create_figure_2) add the prefix
        ax.set_title(title.replace("Ground Truth for", "Ground Truth"), fontsize=12)

        # NOTE: plt.tight_layout() and plt.show() are removed to allow integration into subplots.


def plot_ellipse(ax, mean, cov, color, alpha=0.7, n_std=1.5):
    """
    Draws an ellipse representing a multivariate normal distribution at an n_std confidence level.
    """
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eig_vals)
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        color=color,
        alpha=alpha,
        fill=True,
        linewidth=0,
        zorder=2
    )
    ax.add_patch(ellipse)
    return ellipse


def run_kmeans(X_for_kmeans, k, temporal_weight, genetic_weights):
    """
    Performs K-means clustering on the combined genetic (PCA) and time data.

    Args:
        X_for_kmeans (np.ndarray): Data matrix, where the first columns are
                                   genetic features (PCA) and the last column is time (YBP).
        k (int): The number of clusters.
        temporal_weight (float): The weight multiplier for the time dimension (last column).
        genetic_weights (np.ndarray): Weights for the genetic (PCA) dimensions.

    Returns:
        tuple: (clusters, centroids) where clusters is the array of labels,
               and centroids is the array of cluster centers.
    """
    n_samples, n_features = X_for_kmeans.shape

    # 1. Apply weights to the data
    X_weighted = X_for_kmeans.copy()

    # Apply genetic weights
    if genetic_weights.size > 0:
        X_weighted[:, :-1] *= genetic_weights[:X_weighted.shape[1] - 1]

    # Apply temporal weight
    X_weighted[:, -1] *= temporal_weight

    # 2. Initialize centroids (using k-means++)
    from sklearn.cluster import KMeans

    # Set a specific random state for reproducibility in this example
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')

    # 3. Perform clustering
    kmeans.fit(X_weighted)

    return kmeans.labels_, kmeans.cluster_centers_[:, :-1]  # Centroids in genetic space (excluding weighted time)


def plot_ellipse(ax, mean, cov, color, alpha=0.7, n_std=1.5):
    """
    Draws an ellipse representing a multivariate normal distribution at an n_std confidence level.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the plot on.
        mean (np.ndarray): The mean vector [mean_time, mean_pc1].
        cov (np.ndarray): The covariance matrix.
        color (str): Color of the ellipse.
        alpha (float): Transparency level.
        n_std (float): Number of standard deviations for the ellipse size.
    """
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eig_vals)
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        color=color,
        alpha=alpha,
        fill=True,
        linewidth=0,
        zorder=2
    )
    ax.add_patch(ellipse)
    return ellipse


def _draw_migration_labels(
        axes,
        filtered_lines,
        pop_to_colors,
        AXES_LABEL_Y_BASE=-0.3,
        STACK_HEIGHT_REL=0.12,
        LABEL_TIME_WIDTH_YBP=800,
        MAX_STACK_TRACKS=6,
        is_pc2=False
):
    """
    Draws migration event labels (text and connection lines) at the bottom of the plot.
    Uses Axes Coordinates for the Y position of the text to prevent interference
    with the Data Coordinates (PC1 values).

    Args:
        axes (matplotlib.axes.Axes): The axes object to draw the plot on.
        filtered_lines (list): List of dictionaries containing migration line data.
        pop_to_colors (dict): Dictionary mapping population ID to color list.
        AXES_LABEL_Y_BASE (float): Starting relative Y position (axes coords, typically negative).
        STACK_HEIGHT_REL (float): Relative height increment for stacking labels.
        LABEL_TIME_WIDTH_YBP (int): Time separation required between stacked labels (in YBP).
        MAX_STACK_TRACKS (int): Maximum number of stacking tracks.
        is_pc2 (bool): If True, uses PC2 position for line reference; otherwise uses PC1.
    """
    if not filtered_lines:
        return

    # 1. Sort events by time (YBP, descending: Oldest/Largest YBP first)
    lines_sorted = sorted(filtered_lines, key=lambda x: x['year'], reverse=True)

    # 2. Initialize stacking tracks: stores the 'year' (YBP) of the last label placed on track i
    tracks = [0] * MAX_STACK_TRACKS

    # 3. Iterate and Plot
    for event_data in lines_sorted:
        mig_year = event_data['year']
        # Note: pop_to_colors is not strictly used here for color, but for label logic
        color = event_data['color']
        score_text = event_data['score_text']
        src_pop = event_data['src_pop']
        tgt_pop = event_data['tgt_pop']

        # Find the first available track
        track_index = -1
        for i in range(MAX_STACK_TRACKS):
            # Check if current label is far enough from the last label on this track
            if tracks[i] - mig_year > LABEL_TIME_WIDTH_YBP:
                track_index = i
                break

        if track_index == -1:
            continue

        # Update the track year (set the boundary to the current label's year)
        tracks[track_index] = mig_year

        # Calculate Y position in axes coordinates (relative to bottom of axes, 0 is bottom edge)
        y_label_pos_rel = AXES_LABEL_Y_BASE + (track_index * STACK_HEIGHT_REL)

        # 1. Draw the connection line (from the y_pos_pc1_plot to the bottom edge of the plot area)
        y_pos_key = 'y_pos_pc2_plot' if is_pc2 else 'y_pos_pc1_plot'
        y_plot_ref = event_data.get(y_pos_key)

        y_data_pos_bottom_edge = axes.get_ylim()[0]  # The lowest Data Y value in the current view

        if y_plot_ref is not None:
            # Draw dashed line connecting the event point to the bottom edge of the data view
            axes.plot([mig_year, mig_year],
                      [y_plot_ref, y_data_pos_bottom_edge],
                      color=color,
                      linestyle=':',
                      linewidth=1,
                      alpha=0.6,
                      zorder=1)

        # 2. Draw the Score/Source Label using TRANSFORM
        # This is the FIX: X is Data, Y is Axes
        axes.text(
            mig_year,  # X is Data Coordinate (Time)
            y_label_pos_rel,  # Y is Axes Coordinate (Relative to bottom of axes)
            f'Pop {src_pop + 1} → Pop {tgt_pop + 1}\n({score_text})',
            fontsize=8,
            ha='center',
            va='center',
            color=color,
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle="round,pad=0.3", alpha=0.9),
            # Key change: Use the transformation that combines X Data and Y Axes coordinates
            transform=axes.get_xaxis_transform(),
            zorder=3
        )


def plot_kmeans(
        ax,
        X,
        dates_array,
        pop_clusters,
        temporal_weight=1.0,
        genetic_weights=np.array([1.0, 1.0]),
        k=5,
        migration_lines=None
):
    """
    Performs K-means clustering on weighted genetic (PC1, PC2) and time data,
    and plots the PC1 vs Time results (ellipses, points, and migration labels).
    ... (Args documentation remains the same) ...
    """
    # 1. === Run K-means Clustering ===
    num_genetic_features = len(genetic_weights)
    X_sliced = X[:, :num_genetic_features]

    # --- 🔑 תיקון קריטי: נורמליזציה של ציר הזמן ---
    dates_mean = np.mean(dates_array)
    dates_std = np.std(dates_array)

    # Apply Z-Score Normalization to the time dimension
    if dates_std == 0:
        dates_normalized = dates_array - dates_mean
    else:
        dates_normalized = (dates_array - dates_mean) / dates_std

    # Combine the SLICED PCA data and NORMALIZED time data.
    X_for_kmeans = np.column_stack([X_sliced, dates_normalized])
    # ---------------------------------------------------

    clusters, centroids = run_kmeans(X_for_kmeans, k=k, temporal_weight=temporal_weight,
                                     genetic_weights=genetic_weights)

    pc1 = X[:, 0]

    # 2. === Cluster Coloring Logic ===

    # Step 1: Map each cluster to its dominant population
    cluster_to_pop = {}
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            dominant_pop = Counter(pop_clusters[mask]).most_common(1)[0][0]
            cluster_to_pop[cluster_id] = dominant_pop
        else:
            cluster_to_pop[cluster_id] = -1

    # Step 2: Define colors for each population (using the same palette as the original code)
    pop_to_colors = {
        0: ['#c6dbef', '#6baed6', '#08306b', '#9ecae1', '#4292c6', '#2171b5'],  # Blue
        1: ['#feedde', '#fdae6b', '#d94801', '#fdd0a2', '#fd8d3c', '#f16913'],  # Orange
        2: ['#a1d99b', '#41ab5d', '#00441b', '#74c476', '#238b45', '#1b7837'],  # Green
        3: ['#fcae91', '#ef3b2c', '#67000d', '#fb6a4a', '#cb181d', '#a50f15'],  # Red
        4: ['#bcbddc', '#807dba', '#3f007d', '#9e9ac8', '#6a51a3', '#54278f'],  # Purple
        5: ['#d4dde6', '#6c8597', '#102c3f', '#a1b3c4', '#47697f', '#2c4a5f'],  # Gray-Blue
        6: ['#fccde5', '#f768a1', '#7a0177', '#fa9fb5', '#dd3497', '#ae017e'],  # Pink
    }

    # Step 3 & 4: Calculate average date and assign colors (lighter colors for earlier clusters)
    cluster_avg_dates = []
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            # NOTE: We use the original dates_array here for plotting and date sorting
            mean_date = np.mean(dates_array[mask])
            cluster_avg_dates.append((cluster_id, mean_date))
    cluster_avg_dates.sort(key=lambda x: x[1], reverse=True)  # Sort by date descending (older first)

    cluster_colors = {}
    pop_color_usage = defaultdict(int)
    for _, (cluster_id, _) in enumerate(cluster_avg_dates):
        dominant_pop = cluster_to_pop[cluster_id]
        if dominant_pop != -1:
            colors_list = pop_to_colors[dominant_pop]
            color_index = pop_color_usage[dominant_pop] % len(colors_list)
            cluster_colors[cluster_id] = colors_list[color_index]
            pop_color_usage[dominant_pop] += 1
        else:
            cluster_colors[cluster_id] = '#808080'

    # 3. === Plotting PC1 vs Time ===

    for cluster_id in range(k):
        mask = clusters == cluster_id
        cluster_color = cluster_colors[cluster_id]

        if np.sum(mask) <= 1:
            if np.sum(mask) == 1:
                ax.scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20, zorder=1)
            continue

        # Drawing Ellipse on (Time, PC1) plane
        # Use the original dates_array here for visualization purposes (YBP on X-axis)
        cluster_data_time_pc1 = np.stack((dates_array[mask], pc1[mask]), axis=-1)
        mean_time_pc1 = np.mean(cluster_data_time_pc1, axis=0)
        cov_time_pc1 = np.cov(cluster_data_time_pc1, rowvar=False)

        plot_ellipse(ax, mean_time_pc1, cov_time_pc1, cluster_color, alpha=0.7, n_std=1.5)

        # Drawing Scatter Points
        ax.scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20,
                   zorder=3)  # Increased zorder for points visibility

    # 4. === Plotting Migration Event Labels ===
    # Assuming the call to _draw_migration_labels goes here

    # 5. === Finalizing Axes Settings ===

    ax.set_title(f'KMeans Clustering (k={k}, t={temporal_weight})', fontsize=12)
    ax.set_xlabel('Years Before Present (YBP)')
    ax.set_ylabel('PC1')
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)

def generate_ari_scores(num_of_simulations, num_workers):
    """
    Runs many simulations in parallel and aggregates the ARI scores (mean and std).

    Returns:
        dict: A dictionary containing the processed mean/std data keyed by threshold.
              {th: {'mean_t': np.ndarray, 'std_t': np.ndarray,
                    'mean_k': np.ndarray, 'std_k': np.ndarray,
                    'mean_num_of_subpops': float}}
    """

    scores_for_k = []
    scores_for_t = []
    subpopulations_numbers = []

    # --- Run simulations ---
    # NOTE: Assuming run_simulation is defined elsewhere and accessible
    if __name__ == "__main__":
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # You must ensure 'run_simulation' is defined and returns (sim_scores_for_k, sim_scores_for_t, sim_number_of_subpopulations)
            results = list(executor.map(run_simulation, range(num_of_simulations)))
    else:
        # Handle case where function is called outside of __main__ (e.g., in a notebook or imported script)
        # This part needs a robust way to handle multi-processing initiation if required,
        # but for simplicity, we assume the user ensures 'run_simulation' can be called.
        # For now, we'll return an error or placeholder if outside __main__.
        print("Warning: Running simulations outside __main__ might cause issues.")
        # Alternatively, implement a sequential run or simpler parallelisation:
        # results = [run_simulation(i) for i in range(num_of_simulations)]

    # --- Aggregate results ---
    for sim_scores_for_k, sim_scores_for_t, sim_number_of_subpopulations in results:
        scores_for_k.append(sim_scores_for_k)
        scores_for_t.append(sim_scores_for_t)
        subpopulations_numbers.append(sim_number_of_subpopulations)

    if not scores_for_k:
        print("Error: No simulation results available after aggregation.")
        return {}

    # Define parameter ranges (used for axis alignment later)
    temporal_weight_values_for_k = np.arange(0, 1.5, 0.05)
    k_values_for_k = [5, 10, 20, 30, 40, 100]

    processed_data = {}
    thresholds = list(scores_for_k[0].keys())

    # --- Process data per threshold ---
    for th in thresholds:
        th_scores_for_t = [sim[th] for sim in scores_for_t]
        th_scores_for_t = np.array(th_scores_for_t)

        th_scores_for_k = [sim[th] for sim in scores_for_k]
        th_scores_for_k = np.array(th_scores_for_k)

        th_subpopulations_numbers = [sim[th] for sim in subpopulations_numbers]
        th_subpopulations_numbers = np.array(th_subpopulations_numbers)
        mean_num_of_subpops = th_subpopulations_numbers.mean()

        # Data for ARI vs K (Varying T) plot
        mean_t = th_scores_for_t.mean(axis=0)
        std_t = th_scores_for_t.std(axis=0)

        # Data for ARI vs T (Varying K) plot
        mean_k = th_scores_for_k.mean(axis=0)
        std_k = th_scores_for_k.std(axis=0)

        processed_data[th] = {
            'mean_t': mean_t,
            'std_t': std_t,
            'mean_k': mean_k,
            'std_k': std_k,
            'mean_num_of_subpops': mean_num_of_subpops
        }

    return processed_data


def plot_ARI(ax, threshold, ARI_data):
    """
    Plots ARI vs K for a specific threshold, showing mean and min/max SD as Error Bars.

    Args:
        threshold (float): The migration threshold to plot (e.g., 0, 0.004, 0.008).
        ax (matplotlib.axes.Axes): The axes object to draw the plot on.
        ARI_data (dict): The processed ARI data returned by generate_ari_scores.
    """

    # Define parameter ranges used for plotting (must match ranges in generate_ari_scores)
    temporal_weight_values_for_t = [0, 0.25, 0.5, 1, 10]
    k_values_for_t = np.arange(1, 41, 1)

    # 1. --- Extract Data for the Specific Threshold ---

    if threshold not in ARI_data:
        ax.text(0.5, 0.5, f"No ARI data found for threshold {threshold}",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"D. ARI vs K (Threshold {threshold})", fontsize=12)
        return

    data = ARI_data[threshold]
    mean_t = data['mean_t']
    std_t = data['std_t']
    mean_num_of_subpops = data['mean_num_of_subpops']

    # 2. --- Plotting Setup ---

    ax.set_ylim([0, 1])
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('ARI')

    # 3. --- Draw Lines and Error Bars ---

    for i, t_val in enumerate(temporal_weight_values_for_t):
        mean_vals = mean_t[i]
        std_vals = std_t[i]

        # Draw the mean line
        line, = ax.plot(k_values_for_t, mean_vals, label=f't={t_val}')
        color = line.get_color()

        # --- Draw SD as Vertical Error Bars (Min/Max) ---

        # 1. Find Min SD
        min_std_index = np.argmin(std_vals)
        min_std_k = k_values_for_t[min_std_index]
        min_std_ari_mean = mean_vals[min_std_index]
        min_std_val = std_vals[min_std_index]

        # Draw vertical line for Min SD (Error Bar)
        ax.vlines(x=min_std_k,
                  ymin=min_std_ari_mean - min_std_val,
                  ymax=min_std_ari_mean + min_std_val,
                  color='k', linestyle='-', linewidth=2, zorder=5)
        # Draw caps
        ax.hlines(y=min_std_ari_mean - min_std_val, xmin=min_std_k - 0.5, xmax=min_std_k + 0.5, color='k',
                  linewidth=2, zorder=5)
        ax.hlines(y=min_std_ari_mean + min_std_val, xmin=min_std_k - 0.5, xmax=min_std_k + 0.5, color='k',
                  linewidth=2, zorder=5)

        # Add label for Min SD
        ax.text(min_std_k, min_std_ari_mean + min_std_val + 0.03,
                f'{min_std_val:.3f}',
                color='k', fontsize=7, ha='center',
                bbox=dict(facecolor=color, alpha=0.3, edgecolor='none', boxstyle="round,pad=0.2"))

        # 2. Find Max SD
        max_std_index = np.argmax(std_vals)
        max_std_k = k_values_for_t[max_std_index]
        max_std_ari_mean = mean_vals[max_std_index]
        max_std_val = std_vals[max_std_index]

        # Draw vertical line for Max SD (Error Bar)
        ax.vlines(x=max_std_k,
                  ymin=max_std_ari_mean - max_std_val,
                  ymax=max_std_ari_mean + max_std_val,
                  color='k', linestyle='-', linewidth=2, zorder=5)
        # Draw caps
        ax.hlines(y=max_std_ari_mean - max_std_val, xmin=max_std_k - 0.5, xmax=max_std_k + 0.5, color='k',
                  linewidth=2, zorder=5)
        ax.hlines(y=max_std_ari_mean + max_std_val, xmin=max_std_k - 0.5, xmax=max_std_k + 0.5, color='k',
                  linewidth=2, zorder=5)

        # Add label for Max SD
        ax.text(max_std_k, max_std_ari_mean + max_std_val + 0.03,
                f'{max_std_val:.3f}',
                color='k', fontsize=7, ha='center',
                bbox=dict(facecolor=color, alpha=0.3, edgecolor='none', boxstyle="round,pad=0.2"))

    # 4. --- Final Aesthetic Settings ---

    # Check current position for correct labeling (D, E, F)
    # This requires external context, but we use D. as a default and rely on the caller to manage the label
    title_prefix = {0: "D", 0.004: "E", 0.008: "F"}.get(threshold, "D")
    ax.set_title(f"{title_prefix}. ARI vs K (Threshold {threshold})", fontsize=12)

    # Average subpopulation count annotation
    ax.text(
        0.02, 0.95,
        f"Avg number of populations:\n{mean_num_of_subpops:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )
    ax.legend(loc='upper right', fontsize=8)

    # NOTE: The caller (create_figure_2) is responsible for applying plt.tight_layout() and plt.show()


create_figure_2()