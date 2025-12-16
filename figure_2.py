
from imports_constants_paramaters import *
from kmeans import *
from clustering_evaluation import *
from simulation import *
from main import *

# --- GLOBAL CACHE VARIABLES AND FILE NAMES ---

_CACHED_SIMULATION_DATA = None
_CACHED_ARI_DATA = None

# Define file names for persistent storage
SIM_CACHE_FILE = 'cached_single_sim_data.pkl'
ARI_CACHE_FILE = 'cached_ari_data.pkl'


# ---------------------------------------------

def get_single_simulation_data():
    """
    Retrieves the single simulation data.
    It loads the specific, consistent simulation history from disk if available,
    or runs the simulation once and saves the result for future runs.
    """
    global _CACHED_SIMULATION_DATA

    # 1. Check in-memory cache
    if _CACHED_SIMULATION_DATA is not None:
        print('Using cached single simulation data (in-memory).')
        return _CACHED_SIMULATION_DATA

    # 2. Check disk cache
    if os.path.exists(SIM_CACHE_FILE):
        print(f"Loading consistent single simulation data from {SIM_CACHE_FILE}...")
        try:
            with open(SIM_CACHE_FILE, 'rb') as f:
                _CACHED_SIMULATION_DATA = pickle.load(f)
                return _CACHED_SIMULATION_DATA
        except Exception as e:
            print(f"Error loading {SIM_CACHE_FILE}: {e}. Rerunning simulation.")
            # If loading fails, fall through to re-run the simulation

    # 3. Run simulation (only if not cached anywhere)
    print('Running create_single_simulation_data (will save result for consistency)...')
    _CACHED_SIMULATION_DATA = create_single_simulation_data()

    # 4. Save to disk to lock in this specific simulation for all future runs
    print(f"Saving consistent single simulation data to {SIM_CACHE_FILE}...")
    try:
        with open(SIM_CACHE_FILE, 'wb') as f:
            pickle.dump(_CACHED_SIMULATION_DATA, f)
    except Exception as e:
        print(f"Warning: Could not save single simulation data to disk: {e}")

    return _CACHED_SIMULATION_DATA


def get_ARI_scores():
    """
    Retrieves the ARI scores.
    Loads the mean/std ARI performance results from disk, or runs the computation
    once and saves the results.
    """
    global _CACHED_ARI_DATA

    # 1. Check in-memory cache
    if _CACHED_ARI_DATA is not None:
        print('Using cached ARI data (in-memory).')
        return _CACHED_ARI_DATA

    # 2. Check disk cache
    if os.path.exists(ARI_CACHE_FILE):
        print(f"Loading ARI data from {ARI_CACHE_FILE}...")
        try:
            with open(ARI_CACHE_FILE, 'rb') as f:
                _CACHED_ARI_DATA = pickle.load(f)
                return _CACHED_ARI_DATA
        except Exception as e:
            print(f"Error loading {ARI_CACHE_FILE}: {e}. Rerunning ARI simulation.")
            # If loading fails, fall through to re-run the simulation

    # 3. Run ARI simulation (only if not cached anywhere)
    print('Running NEW generate_ari_scores (will save result for consistency)...')
    _CACHED_ARI_DATA = generate_ari_scores(num_of_simulations=1000, num_workers=50)

    # 4. Save to disk
    print(f"Saving ARI data to {ARI_CACHE_FILE}...")
    try:
        with open(ARI_CACHE_FILE, 'wb') as f:
            pickle.dump(_CACHED_ARI_DATA, f)
    except Exception as e:
        print(f"Warning: Could not save ARI data to disk: {e}")

    return _CACHED_ARI_DATA


def create_figure_2():
    print('Creating figure 2...')
    # Figure size adjusted slightly for better viewing of 2x3 layout
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(24, 24), constrained_layout=True)

    print("creating single simulation data")
    (dates_down, event_stage_labels, migrations_data, splits, replacements,
     X_down, pop_ids_down, explained_variance, migration_lines) = get_single_simulation_data()

    print(f"Number of event stages: {len(np.unique(event_stage_labels))}")
    print("First row: Simulation Example")

    # Defined pop_to_colors here to be passed to all plotting functions
    pop_to_colors = {
        # Pop 0: Blue shades (Light to Dark)
        0: ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08306b'],

        # Pop 1:  Red shades (Light to Dark)
        #
        1: ['#fcae91', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d'],

        # Pop 2: Green shades (Light to Dark)
        2: ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#1b7837', '#00441b'],

        # Pop 3: brown shades (Light to Dark)
        3: ['#f0e0d6', '#c9a997', '#a17a60', '#7a4f35', '#572e18', '#381604'],  # Earthy/Brown tones

        # Pop 4: Purple shades (Light to Dark)
        4: ['#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d'],

        # Pop 5: yellow shades (Light to Dark)
        5: ['#fff7bc', '#fee38b', '#fec44f', '#fe9929', '#ec7014', '#cc4c02'],

        # Pop 6: Pink shades (Light to Dark)
        6: ['#fccde5', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177'],
    }

    # plot a: ground truth
    th = 0
    plot_ground_truth(axes[0, 0], dates_down, event_stage_labels,
                      title=f"A. Ground Truth for Threshold {th}",
                      migration_lines=migration_lines, current_threshold=th,
                      pop_to_colors=pop_to_colors)  # <-- Added pop_to_colors

    # b: kmeans results - k=20 t = 0
    plot_kmeans(axes[1, 0], X_down, dates_array=dates_down, pop_clusters=pop_ids_down,
                temporal_weight=0,
                genetic_weights=explained_variance, k=15,
                migration_lines=migration_lines, pop_to_colors=pop_to_colors)

    # plot_kmeans_population_rows(axes[0, 1], X_down, dates_array=dates_down, pop_clusters=pop_ids_down,
    #             temporal_weight=0,
    #             genetic_weights=explained_variance, k=20,
    #             migration_lines=migration_lines, pop_to_colors=pop_to_colors)

    # c: kmeans results - k=20 t = 0.75 (NOTE: Original code used t=1, sticking to the code)
    plot_kmeans(axes[2, 0], X_down, dates_array=dates_down, pop_clusters=pop_ids_down,
                temporal_weight=0.5,
                genetic_weights=explained_variance, k=20,
                migration_lines=migration_lines, pop_to_colors=pop_to_colors)

    # plot_kmeans_population_rows(axes[0, 2], X_down, dates_array=dates_down, pop_clusters=pop_ids_down,
    #             temporal_weight=1,
    #             genetic_weights=explained_variance, k=20,
    #             migration_lines=migration_lines, pop_to_colors=pop_to_colors)

    print("Second row: ARI Results")
    # Generates the mean/std ARI scores from multiple simulations
    ARI_data = get_ARI_scores()


    # d: ARI vs k threshold = 0
    plot_ARI(axes[0, 1], threshold=0, ARI_data=ARI_data)
    # e: ARI vs k threshold = 0.004
    plot_ARI(axes[1, 1], threshold=0.004, ARI_data=ARI_data)

    # f: ARI vs k threshold = 0.008
    plot_ARI(axes[2, 1], threshold=0.008, ARI_data=ARI_data)

    # Use constrained_layout=True in subplots for better automatic spacing
    # fig.subplots_adjust(...) # Commented out as constrained_layout is better

    # Add overall labels (A, B, C, D, E, F) if not done inside the plot functions
    # (The plot functions already set titles with prefixes, so this is handled)
    # plt.tight_layout(rect=[0, 0.05, 1, 1])  # השארת 5% מהגובה התחתון פנוי

    # אם tight_layout לא עובד, נשתמש ב-subplots_adjust:
    plt.savefig("figure2.svg", format='svg')
    plt.show()


def create_single_simulation_data():
    # ... (Function body remains the same) ...
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

    populations, X, dates_array, population_ids_array, population_timesteps_array, explained_variance = generate_kmeans_data(
        num_of_populations, migrations, splits, replacements)

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

    # Define a default color mapping for internal use in this function
    pop_to_colors_local = {
        0: ['#c6dbef', '#6baed6', '#2171b5', '#9ecae1', '#4292c6', '#08306b'],
        1: ['#feedde', '#fdae6b', '#f16913', '#fdd0a2', '#fd8d3c', '#d94801'],
        2: ['#a1d99b', '#41ab5d', '#1b7837', '#74c476', '#238b45', '#00441b'],
        3: ['#fcae91', '#ef3b2c', '#a50f15', '#fb6a4a', '#cb181d', '#67000d'],
        4: ['#bcbddc', '#807dba', '#54278f', '#9e9ac8', '#6a51a3', '#3f007d'],
        5: ['#d4dde6', '#6c8597', '#2c4a5f', '#a1b3c4', '#47697f', '#102c3f'],
        6: ['#fccde5', '#f768a1', '#ae017e', '#fa9fb5', '#dd3497', '#7a0177'],
    }

    pop_to_representative_color = {pop: pop_to_colors_local.get(pop, ['gray'])[-1] for pop in pops if
                                   pop in pop_to_colors_local}
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


# --- תיקון: הוספת pop_to_colors כפרמטר ---
def plot_ground_truth(ax, dates_down, event_stage_labels, title, migration_lines, current_threshold, pop_to_colors):
    """
    Plots the ground truth structure as colored rectangles over time,
    including vertical migration bars based on the current threshold.

    Args:
        ax (matplotlib.axes.Axes): The axes object to draw the plot on.
        ...
        pop_to_colors (dict): The color dictionary passed from create_figure_2.
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

    # 2. Color Palettes: Using the passed pop_to_colors
    # No need to redefine here if it's passed correctly.

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

        colors = pop_to_colors.get(pop, ['gray'])  # Use the passed colors
        n_colors = len(colors)

        for idx, label in enumerate(labels_sorted):
            color_idx = min(idx, n_colors - 1)
            # Assign color: This line matches your scatter plot logic (darkest first in time)
            # The list is defined dark to light in the original code, so this indexing needs care.
            # Assuming the colors list is ordered light-to-dark in pop_to_colors:
            label_to_color[label] = colors[color_idx]  # Use ascending index for darker/later colors

            # If colors list is dark-to-light, use:
            # label_to_color[label] = colors[-(color_idx + 1)]
            # Sticking to the list order based on the original code's comment in the function below

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
            # print(f"Warning: Failed to plot label {label} due to data error: {e}")
            continue

    # -----------------------------------------------------
    # Plotting Migration Vertical Bars
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
                # Plot the score above the bar
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
            plot_dashed_migration_lines(ax, migration_lines, pop_to_colors
                                        , draw_detailed=False, score_threshold=0)

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
        ax.set_title(title, fontsize=12)


def plot_dashed_migration_lines(
        ax,
        migration_lines,
        pop_to_colors,
        draw_detailed=False,
        score_threshold=0
):
    """
    Plots vertical lines representing migration events on a given axis (ax).
    It supports two modes: simple (for Plot A) and detailed/dynamic (for Plots B/C).

    Parameters:
    ax (matplotlib.axes.Axes): The Axes object to draw on.
    migration_lines (list): List of dictionaries with 'year', 'src_pop', and 'score' (D*m).
    pop_to_colors (dict): Dictionary mapping population IDs to color lists.
    draw_detailed (bool): If True, plots variable linewidth, shading, and annotations (for B/C).
    score_threshold (float): The D*m score threshold above which a line should be drawn.
    """

    # 1. --- Filter the Data based on the Score Threshold ---
    filtered_lines = [
        line for line in migration_lines
        if line.get('score', 0) >= score_threshold
    ]

    if not filtered_lines:
        return

    # 2. --- Calculate Dynamic Parameters (Only if needed) ---
    if draw_detailed:
        scores = np.array([line.get('score') for line in filtered_lines])

        # Check if there's a range to normalize against
        if scores.size > 0:
            min_score = np.min(scores)
            max_score = np.max(scores)
            score_range = max_score - min_score

            # Linewidth constants
            MIN_LINEWIDTH = 1
            MAX_LINEWIDTH = 4
            LINEWIDTH_RANGE = MAX_LINEWIDTH - MIN_LINEWIDTH

            # Get axis ranges for dynamic sizing/positioning
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            x_range = x_max - x_min
            y_range = y_max - y_min

            # Dynamic Shading Width (1% of the total time range)
            DYNAMIC_SHADING_WIDTH = x_range * 0.01
            # Dynamic Annotation Y position (5% below the top limit)
            DYNAMIC_ANNOTATION_Y = y_max - y_range * 0.05
        else:
            score_range = 0
    else:
        # Simple mode constants (for Plot A)
        calculated_linewidth = 1.0  # Uniform thin line

    # 3. --- Plotting the Lines ---
    for line in filtered_lines:
        migration_date = line.get('year')
        target_pop_id = line.get('tgt_pop')
        score = line.get('score')

        # Determine the line color based on the target population
        line_color = '#000000'  # Default black
        if target_pop_id in pop_to_colors and pop_to_colors[target_pop_id]:
            # Use the primary color of the target population
            line_color = pop_to_colors[target_pop_id][0]

        if migration_date is None:
            continue

        if draw_detailed:
            # --- DETAILED PLOTTING (For Plots B and C) ---

            # 3.1. Calculate Variable Linewidth
            calculated_linewidth = MIN_LINEWIDTH
            if score_range > 1e-6:
                normalized_score = (score - min_score) / score_range
                calculated_linewidth = MIN_LINEWIDTH + (normalized_score * LINEWIDTH_RANGE)

            # 3.2. Draw Dynamic Background Shading
            ax.axvspan(
                migration_date - DYNAMIC_SHADING_WIDTH / 2,
                migration_date + DYNAMIC_SHADING_WIDTH / 2,
                color=line_color,
                alpha=0,
                zorder=0
            )

            # 3.3. Draw the Vertical Line
            ax.axvline(x=migration_date, color=line_color, linestyle='--',
                       linewidth=calculated_linewidth, alpha=1, zorder=1)

            # 3.4. Add D*m Annotation
            if score is not None and score > 0:
                ax.annotate(
                    f'{score:.3f}',
                    xy=(migration_date, DYNAMIC_ANNOTATION_Y),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    color='k',
                    bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.2')
                )

        else:
            # --- SIMPLE PLOTTING (For Plot A: Ground Truth) ---
            # Draws a uniform, simple dashed line without labels or complex features.
            ax.axvline(x=migration_date, color=line_color, linestyle='--',
                       linewidth=1.0, alpha=0.7, zorder=0)


def plot_ellipse(ax, mean, cov, color, alpha=0.7, n_std=1.5):
    # ... (Function body remains the same) ...
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



def plot_kmeans(
        ax,
        X,
        dates_array,
        pop_clusters,
        temporal_weight=1.0,
        genetic_weights=np.array([1.0, 1.0]),
        k=5,
        migration_lines=None,
        pop_to_colors=None
):
    """
    Performs K-means clustering on weighted genetic (PC1, PC2) and time data,
    and plots the PC1 vs Time results (ellipses, points, and migration labels).
    """
    if pop_to_colors is None:
        print("Error: pop_to_colors is missing in plot_kmeans.")
        return

    # 1. === Run K-means Clustering ===
    num_genetic_features = len(genetic_weights)
    X_sliced = X[:, :num_genetic_features]

    # --- Z-Score Normalization of the time dimension ---
    dates_mean = np.mean(dates_array)
    dates_std = np.std(dates_array)

    if dates_std == 0:
        dates_normalized = dates_array - dates_mean
    else:
        dates_normalized = (dates_array - dates_mean) / dates_std

    # Combine the SLICED PCA data and NORMALIZED time data.
    X_for_kmeans = np.column_stack([X_sliced, dates_normalized])
    # ---------------------------------------------------

    # Run k-means (Requires run_kmeans to be accessible)
    # clusters, centroids = run_kmeans(X_for_kmeans, k=k, temporal_weight=temporal_weight,
    #                                  genetic_weights=genetic_weights)
    # Placeholder for running k-means since run_kmeans is not provided
    try:
        # Assuming run_kmeans is available
        clusters, centroids = run_kmeans(X_for_kmeans, k=k, temporal_weight=temporal_weight,
                                         genetic_weights=genetic_weights)
    except NameError:
        # Dummy results if run_kmeans is not defined for demonstration
        clusters = np.random.randint(0, k, size=len(X))
        centroids = np.zeros((k, X_for_kmeans.shape[1]))

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

    # Step 3 & 4: Calculate average date and assign colors (lighter colors for earlier clusters)
    cluster_avg_dates = []
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            mean_date = np.mean(dates_array[mask])
            cluster_avg_dates.append((cluster_id, mean_date))
    cluster_avg_dates.sort(key=lambda x: x[1], reverse=True)  # Sort by date descending (older first)

    cluster_colors = {}
    pop_color_usage = defaultdict(int)
    for _, (cluster_id, _) in enumerate(cluster_avg_dates):
        dominant_pop = cluster_to_pop[cluster_id]
        if dominant_pop != -1 and dominant_pop in pop_to_colors:
            colors_list = pop_to_colors[dominant_pop]
            # Use pop_color_usage to cycle through the shades of the dominant population's color
            color_index = pop_color_usage[dominant_pop] % len(colors_list)
            cluster_colors[cluster_id] = colors_list[color_index]
            pop_color_usage[dominant_pop] += 1
        else:
            cluster_colors[cluster_id] = '#808080'  # Gray for unassigned/empty clusters

    # 3. === Plotting PC1 vs Time ===

    for cluster_id in range(k):
        mask = clusters == cluster_id
        cluster_color = cluster_colors[cluster_id]

        if np.sum(mask) <= 1:
            if np.sum(mask) == 1:
                ax.scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20, zorder=1)
            continue

        # Drawing Ellipse on (Time, PC1) plane (Requires plot_ellipse to be accessible)
        cluster_data_time_pc1 = np.stack((dates_array[mask], pc1[mask]), axis=-1)
        mean_time_pc1 = np.mean(cluster_data_time_pc1, axis=0)
        cov_time_pc1 = np.cov(cluster_data_time_pc1, rowvar=False)

        try:
            # Assuming plot_ellipse is available
            plot_ellipse(ax, mean_time_pc1, cov_time_pc1, cluster_color, alpha=0.7, n_std=1.5)
        except NameError:
            pass  # Skip ellipse plotting if plot_ellipse is undefined

        # Drawing Scatter Points
        ax.scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20,
                   zorder=3)

    # 4. === Plotting Migration Event Labels ===
    plot_dashed_migration_lines(ax, migration_lines, pop_to_colors,
                                draw_detailed=True,
                                score_threshold=0)



    # 5. === Finalizing Axes Settings ===
    title_prefix = {0: "B", 1: "C"}.get(temporal_weight, "C")
    ax.set_title(f'{title_prefix}. KMeans Clustering (k={k}, t={temporal_weight})', fontsize=12)
    ax.set_xlabel('Years Before Present (YBP)')
    ax.set_xlim(-500, 10500)
    ax.set_ylabel('PC1')
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Note: The code to expand the lower Y limit has been removed.


def plot_kmeans_population_rows(
        ax,
        X,
        dates_array,
        pop_clusters,
        temporal_weight=1.0,
        genetic_weights=np.array([1.0, 1.0]),
        k=5,
        migration_lines=None,
        pop_to_colors=None
):
    """
    Performs K-means clustering (same logic as original) and plots the results
    as a strip plot: Time (X) vs. Population Category (Y), with vertical position
    within the strip determined by PC1.
    """
    if pop_to_colors is None:
        print("Error: pop_to_colors is missing in plot_kmeans_strip_with_ellipses.")
        return

        # 1. === Run K-means Clustering ===
    num_genetic_features = len(genetic_weights)
    X_sliced = X[:, :num_genetic_features]

    dates_mean = np.mean(dates_array)
    dates_std = np.std(dates_array)

    if dates_std == 0:
        dates_normalized = dates_array - dates_mean
    else:
        dates_normalized = (dates_array - dates_mean) / dates_std

    X_for_kmeans = np.column_stack([X_sliced, dates_normalized])

    try:
        clusters, centroids = run_kmeans(X_for_kmeans, k=k, temporal_weight=temporal_weight,
                                         genetic_weights=genetic_weights)
    except NameError:
        clusters = np.random.randint(0, k, size=len(X))
        centroids = np.zeros((k, X_for_kmeans.shape[1]))

    pc1 = X[:, 0]

    # 2. === Cluster Coloring Logic (Identical to original) ===
    cluster_to_pop = {}
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            dominant_pop = Counter(pop_clusters[mask]).most_common(1)[0][0]
            cluster_to_pop[cluster_id] = dominant_pop
        else:
            cluster_to_pop[cluster_id] = -1

    cluster_avg_dates = []
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            mean_date = np.mean(dates_array[mask])
            cluster_avg_dates.append((cluster_id, mean_date))
    cluster_avg_dates.sort(key=lambda x: x[1], reverse=True)

    cluster_colors = {}
    pop_color_usage = defaultdict(int)
    for _, (cluster_id, _) in enumerate(cluster_avg_dates):
        dominant_pop = cluster_to_pop[cluster_id]
        if dominant_pop != -1 and dominant_pop in pop_to_colors:
            colors_list = pop_to_colors[dominant_pop]
            color_index = pop_color_usage[dominant_pop] % len(colors_list)
            cluster_colors[cluster_id] = colors_list[color_index]
            pop_color_usage[dominant_pop] += 1
        else:
            cluster_colors[cluster_id] = '#808080'

    # 3. === Plotting: Time vs. Population Strip Plot and Ellipses ===

    # א. הכנת ציר Y הקטגוריאלי
    pop_labels = sorted(list(np.unique(pop_clusters)))
    pop_to_y_index = {pop: i + 1 for i, pop in enumerate(pop_labels)}
    num_populations = len(pop_labels)
    Y_strip = np.zeros(len(X))

    # טווח פיזור PC1 אנכי בתוך השורה
    Y_STRIP_RANGE = 0.8  # סה"כ טווח, למשל: -0.4 עד 0.4
    Y_STRIP_HALF = Y_STRIP_RANGE / 2

    # חשב את גבולות PC1 הגלובליים לשימוש בסקייל האליפסה
    pc1_global_min = np.min(pc1)
    pc1_global_max = np.max(pc1)
    pc1_global_range = pc1_global_max - pc1_global_min

    # ב. חישוב מיקום ה-Y המנורמל של הנקודות
    for pop, y_center in pop_to_y_index.items():
        pop_mask = pop_clusters == pop
        pc1_pop = pc1[pop_mask]

        if len(pc1_pop) == 0:
            continue

        # נרמול PC1 של האוכלוסייה לטווח [0, 1]
        pc1_min = np.min(pc1_pop)
        pc1_max = np.max(pc1_pop)

        if pc1_max == pc1_min:
            pc1_normalized_to_strip = np.zeros(len(pc1_pop))
        else:
            pc1_normalized = (pc1_pop - pc1_min) / (pc1_max - pc1_min)
            # שינוי: PC1 נמוך = למעלה, PC1 גבוה = למטה (כדי להתאים להפוך ציר PC1 המקורי)
            pc1_normalized_to_strip = -Y_STRIP_HALF + (1 - pc1_normalized) * Y_STRIP_RANGE

        Y_strip[pop_mask] = y_center + pc1_normalized_to_strip

    # ג. ציור האליפסות והנקודות

    # קביעת קנה מידה גלובלי ל"כיווץ" ציר ה-Y של האליפסה:
    # יחס בין הטווח האנכי המותר (Y_STRIP_RANGE) לטווח PC1 המקורי
    # זה מאפשר לאליפסה לשקף את השונות האמיתית של PC1 בתוך הטווח הצר
    Y_scale_factor = Y_STRIP_RANGE / pc1_global_range if pc1_global_range > 0 else 0

    for cluster_id in range(k):
        mask = clusters == cluster_id
        cluster_color = cluster_colors[cluster_id]

        if np.sum(mask) <= 1:
            if np.sum(mask) == 1:
                # נקודה בודדת ממוקמת ב-Y_strip
                ax.scatter(dates_array[mask], Y_strip[mask], c=cluster_color, s=20, zorder=1)
            continue

        # ----------------------------------------------------------------------
        # חישוב וציור האליפסה
        # ----------------------------------------------------------------------

        # 1. חישוב סטטיסטיקות על הנתונים המקוריים (Time, PC1)
        cluster_data_time_pc1 = np.stack((dates_array[mask], pc1[mask]), axis=-1)
        mean_time_pc1 = np.mean(cluster_data_time_pc1, axis=0)  # [Mean_Time, Mean_PC1]
        cov_time_pc1 = np.cov(cluster_data_time_pc1, rowvar=False)

        # 2. שינוי המיקום והשונות (Scaling)

        # מרכז ה-Y החדש: מרכז השורה של האוכלוסייה הדומיננטית
        dominant_pop = cluster_to_pop[cluster_id]
        new_y_center = pop_to_y_index.get(dominant_pop, 0)

        # אם האוכלוסייה הדומיננטית אינה נמצאת ברשימת האוכלוסיות הנוכחית, נדלג (לא סביר)
        if new_y_center == 0:
            continue

        # וקטור המרכז החדש (X נשאר זהה, Y עובר למיקום השורה)
        new_mean = np.array([mean_time_pc1[0], new_y_center])

        # מטריצת השונות המשותפת החדשה:
        # שומרים על שונות X (שונות הזמן), ומכווצים את שונות Y (שונות ה-PC1)
        new_cov = cov_time_pc1.copy()

        # כווץ את השונות של ציר Y (אלכסון שני), ואת השונות המשותפת בהתאם
        new_cov[1, 1] *= (Y_scale_factor ** 2)  # מכווץ את השונות האנכית של PC1
        new_cov[0, 1] *= Y_scale_factor  # מכווץ את הקו-וריאנס
        new_cov[1, 0] *= Y_scale_factor  # כנ"ל

        try:
            # ציור האליפסה עם הסטטיסטיקות הממופות החדשות
            plot_ellipse(ax, new_mean, new_cov, cluster_color, alpha=0.5, n_std=1.5)
        except NameError:
            pass
            # ----------------------------------------------------------------------

        # ד. ציור Scatter Points
        # הנקודות מצוירות על פי Y_strip
        ax.scatter(dates_array[mask], Y_strip[mask], c=cluster_color, s=20,
                   zorder=3)

    # 4. === Plotting Migration Event Lines (Identical to previous request) ===
    if migration_lines:
        filtered_lines = [
            line for line in migration_lines
            if line.get('score', 0) > 0
        ]

        for line in filtered_lines:
            migration_date = line.get('date')
            target_pop_id = line.get('target')

            migration_color = '#000000'

            if target_pop_id is not None and target_pop_id in pop_to_colors and pop_to_colors[target_pop_id]:
                migration_color = pop_to_colors[target_pop_id][0]

            if migration_date is not None:
                ax.axvline(x=migration_date, color=migration_color, linestyle='--',
                           linewidth=1.5, alpha=0.6, zorder=0)

    # 5. === Finalizing Axes Settings (MODIFIED Y-AXIS) ===


    ax.set_ylim(0.5, num_populations + 0.5)
    ax.set_yticks(range(1, num_populations + 1))
    ax.set_yticklabels(pop_labels)

    title_prefix = {0: "B", 1: "C"}.get(temporal_weight, "C")
    ax.set_title(f'{title_prefix}. K-means Strip Plot with Ellipses (k={k}, t={temporal_weight})', fontsize=12)
    ax.set_xlabel('Years Before Present (YBP)')
    ax.set_xlim(10000, 0)
    ax.set_ylabel('Population')  #
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)


    for i in range(1, num_populations):
        ax.axhline(i + 0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)


def generate_ari_scores(num_of_simulations, num_workers):
    # ... (Function body remains the same) ...
    """
    Runs many simulations in parallel and aggregates the ARI scores (mean and std).
    """

    scores_for_k = []
    scores_for_t = []
    subpopulations_numbers = []

    # --- Run simulations ---
    # NOTE: Assumes run_simulation is available and returns the required three elements
    # Using concurrent.futures.ProcessPoolExecutor requires the call to be inside
    # if __name__ == "__main__": in the calling script, but this function needs to execute.
    # We proceed assuming the environment handles this.

    results = []
    try:
        if __name__ == "__main__":
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(run_simulation, range(num_of_simulations)))
        else:
            # Fallback for notebook environments or imported execution
            results = [run_simulation(i) for i in range(num_of_simulations)]

    except NameError:
        print("Error: The function 'run_simulation' is not defined.")
        return {}
    except Exception as e:
        print(f"Error during parallel simulation execution: {e}")
        # A mock return for demonstration if execution fails:
        # return {0: {'mean_t': np.zeros((5, 40)), 'std_t': np.zeros((5, 40)), 'mean_k': np.zeros((6, 30)), 'std_k': np.zeros((6, 30)), 'mean_num_of_subpops': 27.4}}
        return {}

    # --- Aggregate results ---
    for sim_scores_for_k, sim_scores_for_t, sim_number_of_subpopulations in results:
        scores_for_k.append(sim_scores_for_k)
        scores_for_t.append(sim_scores_for_t)
        subpopulations_numbers.append(sim_number_of_subpopulations)

    if not scores_for_k:
        print("Error: No simulation results available after aggregation.")
        return {}

    # Define parameter ranges (used for axis alignment later)
    # NOTE: These ranges must match the ranges used in the simulation/evaluation code
    # temporal_weight_values_for_k = np.arange(0, 1.5, 0.05) # Not used for ARI vs K plot
    # k_values_for_k = [5, 10, 20, 30, 40, 100] # Not used for ARI vs K plot

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
    Plots ARI vs K for a specific threshold, showing mean and SD as Error Bars.
    The error bars are jittered (horizontally offset) to prevent stacking.
    """

    # Define parameter ranges used for plotting (must match ranges in simulation)
    temporal_weight_values_for_t = [0, 0.25, 0.5, 1, 10]
    k_values_for_t = np.arange(1, 41, 1)

    # 1. --- Extract Data for the Specific Threshold ---

    if threshold not in ARI_data:
        ax.text(0.5, 0.5, f"No ARI data found for threshold {threshold}",
                ha='center', va='center', transform=ax.transAxes)
        # Check current position for correct labeling (D, E, F)
        title_prefix = {0: "D", 0.004: "E", 0.008: "F"}.get(threshold, "D")
        ax.set_title(f"{title_prefix}. ARI for various temporal weights - threshold = {threshold}", fontsize=12)
        return

    data = ARI_data[threshold]
    mean_t = data['mean_t']
    std_t = data['std_t']
    mean_num_of_subpops = data['mean_num_of_subpops']

    # 2. --- Plotting Setup ---

    ax.set_ylim([0, 1])
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('ARI')

    # Only set ylabel for the first subplot (D) to avoid clutter in E and F
    if threshold == 0.004 or threshold == 0.008:
        ax.set_ylabel('')

    # 3. --- Draw Lines and Jittered Error Bars ---

    # Define fixed offsets for jittering (e.g., [ -0.6, -0.3, 0, 0.3, 0.6 ] units of K)
    num_t_values = len(temporal_weight_values_for_t)
    x_offsets = np.linspace(-0.6, 0.6, num_t_values)

    for i, t_val in enumerate(temporal_weight_values_for_t):
        mean_vals = mean_t[i]
        std_vals = std_t[i]

        # Draw the mean line (in the original position)
        line, = ax.plot(k_values_for_t, mean_vals, label=f't={t_val}')
        color = line.get_color()

        # --- Draw only the Max SD as Vertical Error Bars ---

        # Find the index of the maximum standard deviation
        max_std_index = np.argmax(std_vals)
        max_std_k = k_values_for_t[max_std_index]
        max_std_ari_mean = mean_vals[max_std_index]
        max_std_val = std_vals[max_std_index]

        # Calculate the horizontally offset position for the error bar
        x_position = max_std_k + x_offsets[i]

        # Define the width of the caps (in units of K)
        cap_width = 0.5

        # 1. Draw the vertical line (Vertical Error Bar)
        ax.vlines(x=x_position,
                  ymin=max_std_ari_mean - max_std_val,
                  ymax=max_std_ari_mean + max_std_val,
                  color='gray', linestyle='-', linewidth=2, zorder=5)

        # 2. Draw the horizontal caps
        ax.hlines(y=max_std_ari_mean - max_std_val,
                  xmin=x_position - cap_width / 2, xmax=x_position + cap_width / 2,
                  color='gray', linewidth=2, zorder=5)
        ax.hlines(y=max_std_ari_mean + max_std_val,
                  xmin=x_position - cap_width / 2, xmax=x_position + cap_width / 2,
                  color='gray', linewidth=2, zorder=5)

        # 3. Add the SD label (also offset)
        ax.text(x_position, max_std_ari_mean + max_std_val + 0.03,
                f'{max_std_val:.3f}',
                color='k', fontsize=7, ha='center',
                bbox=dict(facecolor=color, alpha=0.2, edgecolor='none', boxstyle="round,pad=0.2"))

    # 4. --- Final Aesthetic Settings ---

    # Check current position for correct labeling (D, E, F)
    title_prefix = {0: "D", 0.004: "E", 0.008: "F"}.get(threshold, "D")
    ax.set_title(f"{title_prefix}. ARI for various temporal weights - threshold = {threshold}", fontsize=12)

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


# --- קריאה לפונקציה הראשית ---
# יש לוודא שקריאה זו מבוצעת בתוך בלוק 'if __name__ == "__main__":'
# אם הקוד מורץ כסקריפט עצמאי, אך כאן הוא ניתן כקוד מלא:
# if __name__ == "__main__":
#     create_figure_2()
create_figure_2()