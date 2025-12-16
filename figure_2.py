
from imports_constants_paramaters import *
from kmeans import *
from clustering_evaluation import *
from simulation import *
from main import *

# --- GLOBAL CACHE VARIABLES AND FILE NAMES ---

_CACHED_SIMULATION_DATA = None
_CACHED_ARI_DATA = None

# Define file names for persistent storage
SIM_CACHE_FILE = 'data_cache/cached_single_sim_data.pkl'
ARI_CACHE_FILE = 'data_cache/cached_ari_data.pkl'


# ---------------------------------------------

def get_single_simulation_data(pop_to_colors):
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
    _CACHED_SIMULATION_DATA = create_single_simulation_data(pop_to_colors)

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

    # font sizes
    BASE_FONT_SIZE = 12  # Choose your desired base size

    # Set the global parameters for all subplots
    plt.rcParams['font.size'] = BASE_FONT_SIZE  # General text size
    plt.rcParams['axes.titlesize'] = BASE_FONT_SIZE + 10  # Main subplot titles (e.g., "B. KMeans Clustering")
    plt.rcParams['axes.labelsize'] = BASE_FONT_SIZE + 2  # Axis labels (e.g., "PC1", "YBP")
    plt.rcParams['xtick.labelsize'] = BASE_FONT_SIZE  # X-axis tick values (numbers)
    plt.rcParams['ytick.labelsize'] = BASE_FONT_SIZE  # Y-axis tick values (numbers)
    plt.rcParams['legend.fontsize'] = BASE_FONT_SIZE - 2  # Legend text size


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

    print("creating single simulation data")
    (dates_down, event_stage_labels, migrations_data, splits, replacements,
     X_down, pop_ids_down, explained_variance, migration_lines) = get_single_simulation_data(pop_to_colors)

    print(f"Number of event stages: {len(np.unique(event_stage_labels))}")
    print("First row: Simulation Example")

    # Defined pop_to_colors here to be passed to all plotting functions

    # plot a: ground truth
    th = 0
    plot_ground_truth(axes[0, 0], dates_down, event_stage_labels,
                      title=f"A. Ground Truth for Threshold {th}",
                      migration_lines=migration_lines, current_threshold=th,
                      pop_to_colors=pop_to_colors)  # <-- Added pop_to_colors

    # b: kmeans results - k=20 t = 0
    plot_kmeans(axes[1, 0], X_down, dates_array=dates_down, pop_clusters=pop_ids_down,
                temporal_weight=0,
                genetic_weights=explained_variance, k=20,
                migration_lines=migration_lines, pop_to_colors=pop_to_colors)

    # plot_kmeans_population_rows(axes[0, 1], X_down, dates_array=dates_down, pop_clusters=pop_ids_down,
    #             temporal_weight=0,
    #             genetic_weights=explained_variance, k=20,
    #             migration_lines=migration_lines, pop_to_colors=pop_to_colors)

    # c: kmeans results - k=20 t = 0.75 (NOTE: Original code used t=1, sticking to the code)
    plot_kmeans(axes[2, 0], X_down, dates_array=dates_down, pop_clusters=pop_ids_down,
                temporal_weight=1,
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
    plt.savefig("figures/figure2.svg", format='svg')
    # plt.show()


def create_single_simulation_data(pop_to_colors=None):
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

    ax.tick_params(axis='both',
                   which='major',
                   labelsize=14)
    ax.set_xlabel('Years Before Present (YBP)', fontsize=18)
    ax.invert_xaxis()
    ax.set_ylabel('Population ID (Clustered)', fontsize=18)

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
        ax.set_title(title)


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
    if draw_detailed:
        filtered_lines = migration_lines
    else:
        # Keep the filtering for Plot A (Ground Truth) which filters by D*m score
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
            line_color = pop_to_colors[target_pop_id][1]

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
                       linewidth=1.5, #calculated_linewidth,
                        alpha=1, zorder=1)

            # 3.4. Add D*m Annotation
            if score is not None and score > 0:
                ax.annotate(
                    f'{score:.3f}',
                    xy=(migration_date, DYNAMIC_ANNOTATION_Y),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    color='k',
                    bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.2')
                )

        else:
            # --- SIMPLE PLOTTING (For Plot A: Ground Truth) ---
            # Draws a uniform, simple dashed line without labels or complex features.
            ax.axvline(x=migration_date, color=line_color, linestyle='--',
                       linewidth=1, alpha=0.1, zorder=0)


def plot_localized_migration_markers(
        ax,
        caught_migration_lines,
        pop_to_colors
):
    """
    Plots vertical line segments for caught migration events, localized to the
    mean PC1 position of the affected population within the time window.
    Linewidth is dynamically scaled by the D*m score.

    This function is intended for Plots B and C (K-Means results).
    """

    if not caught_migration_lines:
        return

    scores = np.array([line.get('score') for line in caught_migration_lines])

    # 1. Calculate Dynamic Parameters

    # Linewidth constants
    MIN_LINEWIDTH = 1
    MAX_LINEWIDTH = 4
    LINEWIDTH_RANGE = MAX_LINEWIDTH - MIN_LINEWIDTH

    # Check for score range for normalization
    if scores.size > 0 and np.max(scores) > np.min(scores):
        min_score = np.min(scores)
        max_score = np.max(scores)
        score_range = max_score - min_score
    else:
        # If all scores are identical or there are no scores, use a base line width.
        score_range = 0
        min_score = 0

    # 2. Plotting the Localized Markers
    for line in caught_migration_lines:
        migration_date = line.get('year')
        target_pop_id = line.get('tgt_pop')
        score = line.get('score')

        # Localized Y position data (calculated in plot_kmeans)
        local_y_center = line.get('local_y_center')
        local_y_height = line.get('local_y_height', 1.0)

        if local_y_center is None:
            continue  # Skip if Y position was not calculated

        # Determine the line color
        line_color = '#000000'
        if target_pop_id in pop_to_colors and pop_to_colors[target_pop_id]:
            line_color = pop_to_colors[target_pop_id][1]

        # 2.1. Calculate Variable Linewidth
        calculated_linewidth = MIN_LINEWIDTH
        if score_range > 1e-6:
            normalized_score = (score - min_score) / score_range
            calculated_linewidth = MIN_LINEWIDTH + (normalized_score * LINEWIDTH_RANGE)

        # 2.2. Draw the Vertical LINE SEGMENT (Localized Line)
        y_start = local_y_center - local_y_height / 2
        y_end = local_y_center + local_y_height / 2

        ax.plot(
            [migration_date, migration_date],  # X coordinates are the same (vertical line)
            [y_start, y_end],  # Y coordinates define the segment height
            color=line_color,
            linestyle='--',
            linewidth=calculated_linewidth,
            alpha=1,
            zorder=1
        )

        # 2.3. Add D*m Annotation
        if score is not None and score > 0:
            ax.annotate(
                f'{score:.3f}',
                xy=(migration_date, y_end),  # Anchor at the top of the segment
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                color='k',
                bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.2'),
                zorder=5
            )


def plot_migration_arrows(
        ax,
        caught_migration_lines,
        pop_to_colors
):
    """
    Plots arrows representing caught migration events, running from the
    Source Population's mean PC1 position to the Target Population's mean PC1 position.
    Linewidth is dynamically scaled by the D*m score.
    """

    if not caught_migration_lines:
        return

    scores = np.array([line.get('score') for line in caught_migration_lines])

    # 1. Calculate Dynamic Parameters

    MIN_LINEWIDTH = 1.0
    MAX_LINEWIDTH = 4.0
    LINEWIDTH_RANGE = MAX_LINEWIDTH - MIN_LINEWIDTH
    ARROW_HEAD_WIDTH = 0.5  # Default width for a decent arrow head

    # Check for score range for normalization
    if scores.size > 0 and np.max(scores) > np.min(scores):
        min_score = np.min(scores)
        max_score = np.max(scores)
        score_range = max_score - min_score
    else:
        score_range = 0
        min_score = 0

    # 2. Plotting the Arrows
    for line in caught_migration_lines:
        migration_date = line.get('year')
        target_pop_id = line.get('tgt_pop')
        score = line.get('score')

        # Retrieve calculated Y positions (Source and Target)
        y_src = line.get('local_y_center_src')
        y_tgt = line.get('local_y_center_tgt')

        if y_src is None or y_tgt is None:
            continue  # Skip if Y position was not calculated for both

        # Determine the arrow color (based on Target Pop)
        line_color = '#000000'
        if target_pop_id in pop_to_colors and pop_to_colors[target_pop_id]:
            line_color = pop_to_colors[target_pop_id][1]

        src_pop_color = '#000000'
        if line.get('src_pop') in pop_to_colors and pop_to_colors[line.get('src_pop')]:
            src_pop_color = pop_to_colors[line.get('src_pop')][1]

        # 2.1. Calculate Variable Linewidth (affects arrow thickness)
        calculated_linewidth = MIN_LINEWIDTH
        if score_range > 1e-6:
            normalized_score = (score - min_score) / score_range
            calculated_linewidth = MIN_LINEWIDTH + (normalized_score * LINEWIDTH_RANGE)

        # 2.2. Draw the Arrow
        # dx=0 because it's a vertical arrow, dy is the difference in PC1 means.
        dy = y_tgt - y_src

        # Use ax.arrow for better control over thickness (linewidth)
        # Note: head_width must be relative to the X-axis scale (Time)

        ax.arrow(
            x=migration_date,  # X position
            y=y_src,  # Start Y position
            dx=0,  # No movement in X
            dy=dy,  # Total movement in Y (from src to tgt)
            head_width=200,
            head_length=0.4,
            fc=line_color,
            ec='k',
            linewidth=calculated_linewidth,  # Use linewidth for the body
            length_includes_head=True,
            alpha=1,
            zorder=4
        )

        # 2.3. Add D*m Annotation (positioned near the target end of the arrow)
        # if score is not None and score > 0:
        #     # Place annotation slightly offset from the target point
        #     annotate_y = y_tgt + (0.5 if dy >= 0 else -0.5)

            # ax.annotate(
            #     f'{score:.3f}',
            #     xy=(migration_date, annotate_y),
            #     xytext=(0, 0),
            #     textcoords='offset points',
            #     ha='center',
            #     color='k',
            #     bbox=dict(facecolor='white', alpha=1, edgecolor='none', boxstyle='round,pad=0.2'),
            #     zorder=5
            # )


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


def is_migration_caught(samples_df, migration_time, affected_pop_id, cluster_labels, time_window_size=500,
                        min_dominance_ratio=0.75):
    """
    Checks if a migration event caused a shift in cluster assignments for the AFFECTED population
    by comparing samples within fixed time windows before and after the event.
    """

    df = samples_df.copy()
    df['Cluster'] = cluster_labels

    # 1. Filter ONLY for the affected population in the Ground Truth (Crucial change!)
    df_affected = df[df['Pop_ID'] == affected_pop_id]

    if df_affected.empty:
        return False

    # Define time windows (Time is YBP, larger is older, smaller is newer)
    time_after_start = migration_time - time_window_size  # Newest YBP for the 'after' window
    time_before_end = migration_time + time_window_size  # Oldest YBP for the 'before' window

    MIN_SAMPLES = 5

    # 2. Window BEFORE the event (Older samples: Time > migration_time, but near it)
    before_event_window = df_affected[
        (df_affected['Time'] > migration_time) &
        (df_affected['Time'] <= time_before_end)
        ]

    # 3. Window AFTER the event (Newer samples: Time <= migration_time, but near it)
    after_event_window = df_affected[
        (df_affected['Time'] <= migration_time) &
        (df_affected['Time'] >= time_after_start)
        ]

    # Check for sufficient sample size
    if len(before_event_window) < MIN_SAMPLES or len(after_event_window) < MIN_SAMPLES:
        return False

    # 4. Find the dominant cluster and check dominance ratio

    # Check BEFORE
    cluster_counts_before = Counter(before_event_window['Cluster'])
    dominant_cluster_before = cluster_counts_before.most_common(1)

    # Check AFTER
    cluster_counts_after = Counter(after_event_window['Cluster'])
    dominant_cluster_after = cluster_counts_after.most_common(1)

    if not dominant_cluster_before or not dominant_cluster_after:
        return False

    cluster_id_before = dominant_cluster_before[0][0]
    cluster_id_after = dominant_cluster_after[0][0]

    # Check the ratio of the dominant cluster vs. total samples in the window
    dominance_ratio_before = dominant_cluster_before[0][1] / len(before_event_window)
    dominance_ratio_after = dominant_cluster_after[0][1] / len(after_event_window)

    # 5. Final Check: Shift AND High Dominance

    # The event is caught ONLY IF:
    # a) The dominant cluster ID changed
    # b) Both the 'before' and 'after' clusters meet the minimum dominance requirement
    is_shifted = cluster_id_before != cluster_id_after
    is_dominant = (dominance_ratio_before >= min_dominance_ratio) and \
                  (dominance_ratio_after >= min_dominance_ratio)

    return is_shifted and is_dominant


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
    and plots the PC1 vs Time results (ellipses, points, and migration arrows).
    """
    if pop_to_colors is None:
        print("Error: pop_to_colors is missing in plot_kmeans.")
        return

    # 1. === Run K-means Clustering ===
    num_genetic_features = len(genetic_weights)
    X_sliced = X[:, :num_genetic_features]

    # Z-Score Normalization of the time dimension (ensures time dimension does not dominate based on scale)
    dates_mean = np.mean(dates_array)
    dates_std = np.std(dates_array)
    dates_normalized = (dates_array - dates_mean) / dates_std if dates_std != 0 else dates_array - dates_mean

    # Combine the SLICED PCA data and NORMALIZED time data for clustering.
    X_for_kmeans = np.column_stack([X_sliced, dates_normalized])

    try:
        # Run k-means with specified weights.
        clusters, centroids = run_kmeans(X_for_kmeans, k=k, temporal_weight=temporal_weight,
                                         genetic_weights=genetic_weights)
    except NameError:
        # Fallback for demonstration if run_kmeans is not defined.
        clusters = np.random.randint(0, k, size=len(X))
        centroids = np.zeros((k, X_for_kmeans.shape[1]))

    pc1 = X[:, 0]

    # 2. === Cluster Coloring Logic ===

    # Step 1: Map each cluster to its dominant ground-truth population ID.
    cluster_to_pop = {}
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            dominant_pop = Counter(pop_clusters[mask]).most_common(1)[0][0]
            cluster_to_pop[cluster_id] = dominant_pop
        else:
            cluster_to_pop[cluster_id] = -1

    # Calculate average date for each cluster.
    cluster_avg_dates = []
    for cluster_id in range(k):
        mask = clusters == cluster_id
        if np.sum(mask) > 0:
            mean_date = np.mean(dates_array[mask])
            cluster_avg_dates.append((cluster_id, mean_date))

    # Sort clusters by date descending (older samples first).
    cluster_avg_dates.sort(key=lambda x: x[1], reverse=True)

    # Assign colors: Use darker shades for older clusters within the same dominant population.
    cluster_colors = {}
    pop_color_usage = defaultdict(int)
    for _, (cluster_id, _) in enumerate(cluster_avg_dates):
        dominant_pop = cluster_to_pop[cluster_id]
        if dominant_pop != -1 and dominant_pop in pop_to_colors:
            colors_list = pop_to_colors[dominant_pop]
            # Cycle through the shades based on usage count.
            color_index = pop_color_usage[dominant_pop] % len(colors_list)
            cluster_colors[cluster_id] = colors_list[color_index]
            pop_color_usage[dominant_pop] += 1
        else:
            cluster_colors[cluster_id] = '#808080'  # Gray for unassigned/empty clusters

    # 3. === Plotting PC1 vs Time (Scatter and Ellipses) ===
    for cluster_id in range(k):
        mask = clusters == cluster_id
        cluster_color = cluster_colors[cluster_id]

        if np.sum(mask) <= 1:
            # Draw single point if only one sample.
            if np.sum(mask) == 1:
                ax.scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20, zorder=1)
            continue

        # Prepare data for ellipse (Time vs PC1).
        cluster_data_time_pc1 = np.stack((dates_array[mask], pc1[mask]), axis=-1)
        mean_time_pc1 = np.mean(cluster_data_time_pc1, axis=0)
        cov_time_pc1 = np.cov(cluster_data_time_pc1, rowvar=False)

        try:
            # Draw the confidence ellipse.
            plot_ellipse(ax, mean_time_pc1, cov_time_pc1, cluster_color, alpha=0.7, n_std=1.5)
        except NameError:
            pass  # Skip ellipse plotting if plot_ellipse is undefined

        # Draw Scatter Points.
        ax.scatter(dates_array[mask], pc1[mask], c=cluster_color, s=20, zorder=3)

    # 4. === Plotting Migration Event Labels (Arrows) ===

    if migration_lines is None:
        return

    # 4.1. Prepare the data frame for detection
    samples_df = pd.DataFrame({
        'Time': dates_array,
        'Pop_ID': pop_clusters,  # Ground Truth Pop ID is used as a filter
        'PC1': X[:, 0]
    })

    time_window_size = 1500  # Define the time window size for 'caught' detection

    # Filter only migration lines that have a sufficient score to be considered (score > 0).
    relevant_migrations = [line for line in migration_lines if line.get('score', 0) > 0]

    caught_migration_lines_with_pos = []
    caught_migration_lines=[]

    # Loop through relevant migrations to check if the K-Means clustering 'caught' the event.
    for line_data in relevant_migrations:
        mig_time = line_data.get('year')
        affected_pop_tgt = line_data.get('tgt_pop')  # Target Population ID
        affected_pop_src = line_data.get('src_pop')  # Source Population ID

        # Determine if the K-Means clustering 'caught' this specific temporal change (based on TGT pop).
        try:
            is_caught = is_migration_caught(
                samples_df,
                mig_time,
                affected_pop_tgt,  # Check shift based on TARGET population
                clusters,
                time_window_size=time_window_size,
                min_dominance_ratio=0.75
            )
        except NameError:
            # Fallback if is_migration_caught is not defined.
            print("Error: is_migration_caught function is not defined. Skipping migration filtering.")
            is_caught = True

        if is_caught:
            # --- Calculate Y position for both Source and Target populations ---
            caught_migration_lines.append(line_data)

            # Filter samples of the TARGET population within the time window
            df_tgt_window = samples_df[
                (samples_df['Pop_ID'] == affected_pop_tgt) &
                (samples_df['Time'] <= mig_time + time_window_size) &
                (samples_df['Time'] >= mig_time - time_window_size)
                ]

            # Filter samples of the SOURCE population within the time window
            df_src_window = samples_df[
                (samples_df['Pop_ID'] == affected_pop_src) &
                (samples_df['Time'] <= mig_time + time_window_size) &
                (samples_df['Time'] >= mig_time - time_window_size)
                ]

            # Calculate the average PC1 position (Y center) for both
            if not df_tgt_window.empty and not df_src_window.empty:
                avg_pc1_tgt = df_tgt_window['PC1'].mean()
                avg_pc1_src = df_src_window['PC1'].mean()

                # Create a copy and store the calculated Y positions.
                line_data_with_pos = line_data.copy()
                line_data_with_pos['local_y_center_tgt'] = avg_pc1_tgt
                line_data_with_pos['local_y_center_src'] = avg_pc1_src

                caught_migration_lines_with_pos.append(line_data_with_pos)

    # 4.2. Plot the arrows using the dedicated function.
    plot_migration_arrows(ax, caught_migration_lines_with_pos, pop_to_colors)
    plot_dashed_migration_lines(ax, caught_migration_lines, pop_to_colors, draw_detailed=True, score_threshold=0)

    # 5. === Finalizing Axes Settings ===
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=14)
    title_prefix = {0: "B", 0.5: "C"}.get(temporal_weight, "C")
    ax.set_title(f'{title_prefix}. KMeans Clustering (k={k}, t={temporal_weight})')
    ax.set_xlabel('Years Before Present (YBP)', fontsize=18)
    ax.set_xlim(-500, 10500)
    ax.set_ylabel('PC1', fontsize=18)
    ax.invert_xaxis()
    ax.grid(True, linestyle='--', alpha=0.6)


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
                migration_color = pop_to_colors[target_pop_id][5]

            if migration_date is not None:
                ax.axvline(x=migration_date, color=migration_color, linestyle='--',
                           linewidth=1.5, alpha=0.6, zorder=0)

    # 5. === Finalizing Axes Settings (MODIFIED Y-AXIS) ===


    ax.set_ylim(0.5, num_populations + 0.5)
    ax.set_yticks(range(1, num_populations + 1))
    ax.set_yticklabels(pop_labels)

    title_prefix = {0: "B", 1: "C"}.get(temporal_weight, "C")
    ax.set_title(f'{title_prefix}. K-means Strip Plot with Ellipses (k={k}, t={temporal_weight})')
    ax.set_xlabel('Years Before Present (YBP)', fontsize=18)
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
        ax.set_title(f"{title_prefix}. ARI for various temporal weights - threshold = {threshold}")
        return

    data = ARI_data[threshold]
    mean_t = data['mean_t']
    std_t = data['std_t']
    mean_num_of_subpops = data['mean_num_of_subpops']

    # 2. --- Plotting Setup ---

    ax.set_ylim([0, 1])
    ax.set_xlabel('Number of clusters (K)', fontsize=18)
    ax.set_ylabel('ARI', fontsize=18)

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
        # ax.vlines(x=x_position,
        #           ymin=max_std_ari_mean - max_std_val,
        #           ymax=max_std_ari_mean + max_std_val,
        #           color='gray', linestyle='-', linewidth=2, zorder=5)

        # 2. Draw the horizontal caps
        # ax.hlines(y=max_std_ari_mean - max_std_val,
        #           xmin=x_position - cap_width / 2, xmax=x_position + cap_width / 2,
        #           color='gray', linewidth=2, zorder=5)
        # ax.hlines(y=max_std_ari_mean + max_std_val,
        #           xmin=x_position - cap_width / 2, xmax=x_position + cap_width / 2,
        #           color='gray', linewidth=2, zorder=5)

        # 3. Add the SD label (also offset)
        # ax.text(x_position, max_std_ari_mean + max_std_val + 0.03,
        #         f'{max_std_val:.3f}',
        #         color='k', ha='center',
        #         bbox=dict(facecolor=color, alpha=0.2, edgecolor='none', boxstyle="round,pad=0.2"))

    # 4. --- Final Aesthetic Settings ---

    # Check current position for correct labeling (D, E, F)
    title_prefix = {0: "D", 0.004: "E", 0.008: "F"}.get(threshold, "D")
    ax.set_title(f"{title_prefix}. ARI for various temporal weights - threshold = {threshold}")

    # Average subpopulation count annotation
    ax.text(
        0.02, 0.95,
        f"Avg number of populations:\n{mean_num_of_subpops:.2f}",
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )
    ax.legend(loc='upper right')
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=14)


# --- קריאה לפונקציה הראשית ---
# יש לוודא שקריאה זו מבוצעת בתוך בלוק 'if __name__ == "__main__":'
# אם הקוד מורץ כסקריפט עצמאי, אך כאן הוא ניתן כקוד מלא:
# if __name__ == "__main__":
#     create_figure_2()
create_figure_2()