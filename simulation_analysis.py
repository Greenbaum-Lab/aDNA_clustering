from imports_constants_paramaters import *

def result_pca(populations, result_data_matrix, selective_alleles, migrations, splits, replacements, frequency_threshold=0.01):
    """
    Perform PCA on the allele frequency data matrix (result_data_matrix) and plot PC1 over time and PC1 vs PC2.

    Args:
        populations (list): List of populations with metadata.
        result_data_matrix (np.array): Matrix with population, generation and allele frequency data.
        selective_alleles (list): Indices of alleles to exclude from PCA.
        migrations (list): List of migration events.
        splits (list): List of split events.
        replacements (list): List of replacement events.
        frequency_threshold (float): Threshold to exclude low-frequency alleles.
    """
    num_of_variants = result_data_matrix.shape[1] - 2
    columns = ['population', 'generation'] + [f'variant {i+1}' for i in range(num_of_variants)]
    df = pd.DataFrame(result_data_matrix, columns=columns)

    # Remove selective alleles columns from PCA
    for i in selective_alleles:
        df.drop(columns=['variant ' + str(i + 1)], inplace=True)

    x = df.copy()
    x.drop(columns=['population', 'generation'], inplace=True)

    # Perform PCA with 2 components
    pca = PCA(n_components=2)
    components = pca.fit_transform(x)

    df['PC1'] = components[:, 0]
    df['PC2'] = components[:, 1]
    explained_variance = pca.explained_variance_ratio_

    generation_step = 25
    max_generation = 400

    plt.figure(figsize=(12, 8))

    # Scatter plot of PC1 vs years before present for each population
    for pop in df['population'].unique():
        subset = df[df['population'] == pop].sort_values('generation')
        years_before_present = (max_generation - subset['generation']) * generation_step
        plt.scatter(years_before_present, subset['PC1'], label='Population ' + str(int(pop) + 1) + '-' + populations[int(pop)][AREA])

    # Add event lines for migrations, splits, replacements
    for migration in migrations:
        x = (max_generation - migration[MIG_GEN]) * generation_step
        plt.axvline(x=x, color='gray', linestyle='dashed', label=f'Migration: {migration[SRC_POP] + 1} -> {migration[TGT_POP] + 1} {(max_generation - migration[MIG_GEN]) * generation_step} years ago')

    for split_event in splits:
        x = (max_generation - split_event[SPLIT_GEN]) * generation_step
        plt.axvline(x=x, color='blue', linestyle='dashed', label=f'Split: pop {split_event[SRC_POP] + 1} {(max_generation - split_event[SPLIT_GEN]) * generation_step} years ago')

    for replacement in replacements:
        x = (max_generation - replacement[RPC_GEN]) * generation_step
        plt.axvline(x=x, color='red', linestyle='dashed', label=f'Replacement: pop {replacement[SRC_POP] + 1} replaces {replacement[TGT_POP] + 1} {(max_generation - replacement[RPC_GEN]) * generation_step} years ago')

    plt.title('Frequency PC1 over time')
    plt.grid(True)
    plt.xlabel('Years before present')
    plt.xlim(10000, 0)
    plt.ylabel('PC1 ({}%)'.format(round(explained_variance[0] * 100, 1)))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # PC1 vs PC2 scatter plot
    plt.figure(figsize=(12, 8))

    for pop in df['population'].unique():
        subset = df[df['population'] == pop]
        plt.scatter(subset['PC1'], subset['PC2'], label='Population ' + str(int(pop) + 1) + '-' + populations[int(pop)][AREA])

    plt.title('Frequency PCA results')
    plt.grid(True)
    plt.xlabel('PC1 ({}%)'.format(round(explained_variance[0] * 100, 1)))
    plt.ylabel('PC2({}%)'.format(round(explained_variance[1] * 100, 1)))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def samples_pca(populations, samples, selective_alleles):
    columns = ['population', 'generation'] + [f'variant {i + 1}' for i in range(num_of_variants)]
    df = pd.DataFrame(samples, columns=columns)

    for i in selective_alleles:
        df.drop(columns=['variant ' + str(i + 1)], inplace=True)

    x = df.copy()
    x.drop(columns=['population', 'generation'], inplace=True)
    pca = PCA(n_components=4)
    components = pca.fit_transform(x)

    df['PC1'] = components[:, 0]
    df['PC2'] = components[:, 1]
    df['PC3'] = components[:, 2]
    df['PC4'] = components[:, 3]
    explained_variance = pca.explained_variance_ratio_
    return df, explained_variance


def plot_sample_pca_separately(df, populations, migrations, splits, replacements, explained_variance):
    import matplotlib.pyplot as plt

    generation_step = 25
    max_generation = 400

    def plot_pc_vs_pc(df, explained_variance):
        plt.figure(figsize=(12, 8))
        for pop in df['population'].unique():
            subset = df[df['population'] == pop]
            plt.scatter(subset['PC1'], subset['PC2'],
                        label='Population ' + str(int(pop) + 1) + '-' + populations[int(pop)][AREA])
        plt.title('Samples PCA results (PC1 vs PC2)')
        plt.xlabel(f'PC1 ({round(explained_variance[0] * 100, 1)}%)')
        plt.ylabel(f'PC2 ({round(explained_variance[1] * 100, 1)}%)')
        plt.grid(True)
        plt.legend(title='Legend')
        plt.tight_layout()
        plt.show()

    def plot_pc_over_time(df, pc_label, ylabel_percent, title):
        plt.figure(figsize=(12, 6))
        for pop in df['population'].unique():
            subset = df[df['population'] == pop].sort_values('generation')
            years_before_present = (max_generation - subset['generation']) * generation_step
            plt.scatter(years_before_present, subset[pc_label],
                        label='Population ' + str(int(pop) + 1) + '-' + populations[int(pop)][AREA])

        # for migration in migrations:
        #     x = (max_generation - migration[MIG_GEN]) * generation_step
        #     plt.axvline(x=x, color='gray', linestyle='dashed', label=f'Migration: {migration[SRC_POP]+1} -> {migration[TGT_POP]+1} {x} yrs ago')

        # for split_event in splits:
        #     x = (max_generation - split_event[SPLIT_GEN]) * generation_step
        #     plt.axvline(x=x, color='blue', linestyle='dashed', label=f'Split: pop {split_event[SRC_POP]+1} {x} yrs ago')

        # for replacement in replacements:
        #     x = (max_generation - replacement[RPC_GEN]) * generation_step
        #     plt.axvline(x=x, color='red', linestyle='dashed', label=f'Replacement: pop {replacement[SRC_POP]+1} -> {replacement[TGT_POP]+1} {x} yrs ago')

        plt.title(title)
        plt.xlabel('Years before present')
        plt.ylabel(f'{pc_label} ({round(ylabel_percent * 100, 1)}%)')
        plt.xlim(10000, 0)
        plt.grid(True)
        plt.legend(title='Legend')
        plt.tight_layout()
        plt.show()

    # Call separate plots
    plot_pc_vs_pc(df, explained_variance)  # PC1 vs PC2

    if len(explained_variance) > 0:
        plot_pc_over_time(df, 'PC1', explained_variance[0], 'Samples PC1 over time')
    if len(explained_variance) > 1:
        plot_pc_over_time(df, 'PC2', explained_variance[1], 'Samples PC2 over time')
    if len(explained_variance) > 2 and 'PC3' in df.columns:
        plot_pc_over_time(df, 'PC3', explained_variance[2], 'Samples PC3 over time')
    if len(explained_variance) > 3 and 'PC4' in df.columns:
        plot_pc_over_time(df, 'PC4', explained_variance[3], 'Samples PC4 over time')


def plot_sample_pca(df, populations, migrations, splits, replacements, explained_variance):
    generation_step = 25
    max_generation = 400

    # Create a 2x3 grid of subplots (6 slots, some remain empty)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()  # flatten for easier indexing

    def plot_pc_over_time(ax, pc_label, pc_num, ylabel_percent, title):
        for pop in df['population'].unique():
            subset = df[df['population'] == pop].sort_values('generation')
            years_before_present = (max_generation - subset['generation']) * generation_step
            ax.scatter(years_before_present, subset[pc_label],
                       label='Population ' + str(int(pop) + 1) + '-' + populations[int(pop)][AREA])

        for migration in migrations:
            x = (max_generation - migration[MIG_GEN]) * generation_step
            ax.axvline(x=x, color='gray', linestyle='dashed',
                       label=f'Migration: {migration[SRC_POP] + 1} -> {migration[TGT_POP] + 1} {(max_generation - migration[MIG_GEN]) * generation_step} years ago')

        for split_event in splits:
            x = (max_generation - split_event[SPLIT_GEN]) * generation_step
            ax.axvline(x=x, color='blue', linestyle='dashed',
                       label=f'Split: pop {split_event[SRC_POP] + 1} {(max_generation - split_event[SPLIT_GEN]) * generation_step} years ago')

        for replacement in replacements:
            x = (max_generation - replacement[RPC_GEN]) * generation_step
            ax.axvline(x=x, color='red', linestyle='dashed',
                       label=f'Replacement: pop {replacement[SRC_POP] + 1} replaces {replacement[TGT_POP] + 1} {(max_generation - replacement[RPC_GEN]) * generation_step} years ago')

        ax.set_title(title)
        ax.grid(True)
        ax.set_xlabel('Years before present')
        ax.set_xlim(10000, 0)
        ax.set_ylabel(f'{pc_label} ({round(ylabel_percent * 100, 1)}%)')

    # PC1 vs PC2 scatter plot on the left-most subplot (index 0)
    ax = axes[0]
    for pop in df['population'].unique():
        subset = df[df['population'] == pop]
        ax.scatter(subset['PC1'], subset['PC2'],
                   label='Population ' + str(int(pop) + 1) + '-' + populations[int(pop)][AREA])
    ax.set_title('Samples PCA results')
    ax.grid(True)
    ax.set_xlabel(f'PC1 ({round(explained_variance[0] * 100, 1)}%)')
    ax.set_ylabel(f'PC2 ({round(explained_variance[1] * 100, 1)}%)')

    # PC1 over time in subplot index 1
    plot_pc_over_time(axes[1], 'PC1', 1, explained_variance[0], 'Samples PC1 over time')

    # PC2 over time in subplot index 2
    plot_pc_over_time(axes[2], 'PC2', 2, explained_variance[1], 'Samples PC2 over time')

    # Remove unused last three subplots (indices 3,4,5)
    for i in range(3, 6):
        fig.delaxes(axes[i])

    # Collect all handles and labels from the axes to create a single combined legend
    handles = []
    labels = []
    for ax in axes[:3]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicate entries from the legend
    unique = dict(zip(labels, handles))
    unique_labels = list(unique.keys())
    unique_handles = list(unique.values())

    # Create a single combined legend on the right side of the figure
    fig.legend(unique_handles, unique_labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title='Legend')

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend
    plt.show()


def calculate_FST(pop1, pop2, generation):
    """
    Calculates FST between two populations at a specific generation.

    Parameters:
        pop1 (dict): First population dictionary.
        pop2 (dict): Second population dictionary.
        generation (int): The generation at which to calculate FST.

    Returns:
        float or None: FST value, or None if generation is out of bounds.
    """
    # Skip if the generation is before either population existed
    if pop1[START_GEN] > generation or pop2[START_GEN] > generation:
        return None

    # Get allele frequencies for the given generation
    pop1_freqs = pop1[FREQS][generation - pop1[START_GEN]]
    pop2_freqs = pop2[FREQS][generation - pop2[START_GEN]]

    fst = 0
    for i in range(num_of_variants):
        p1 = pop1_freqs[i]
        p2 = pop2_freqs[i]
        H1 = 2 * p1 * (1 - p1)
        H2 = 2 * p2 * (1 - p2)
        Hs = 0.5 * (H1 + H2)
        p = (p1 + p2) / 2
        Ht = 2 * p * (1 - p)
        if Ht != 0:
            fst += (Ht - Hs) / Ht
    fst /= num_of_variants
    return fst


def calculate_FST_matrix(populations, generation):
    """
    Calculates pairwise FST matrix for all populations at a given generation.

    Parameters:
        populations (list): List of population dictionaries.
        generation (int): Generation at which to calculate FST.

    Returns:
        np.array: Symmetric FST matrix.
    """
    fst_matrix = np.zeros((len(populations), len(populations)))
    for i in range(len(populations)):
        pop1 = populations[i]
        for j in range(len(populations)):
            pop2 = populations[j]
            if i == j:
                fst_matrix[i][j] = 0  # FST with itself is zero
            else:
                fst_matrix[i][j] = calculate_FST(pop1, pop2, generation)
                fst_matrix[j][i] = fst_matrix[i][j]  # Mirror the value
    return fst_matrix


def create_FST_matrices(populations, fst_generations):
    """
    Creates FST matrices for a list of generations.

    Parameters:
        populations (list): List of population dictionaries.
        fst_generations (list): List of generations at which to compute FST.

    Returns:
        list of np.array: FST matrices for each specified generation.
    """
    fst_matrices = []
    for generation in fst_generations:
        generation = 1 if generation == 0 else generation
        fst_matrix = calculate_FST_matrix(populations, generation - 1)
        fst_matrices.append(fst_matrix)
    return fst_matrices


def show_FST_matrices(populations, fst_matrices):
    """
    Displays a heatmap for each FST matrix.

    Parameters:
        populations (list): List of population dictionaries.
        fst_matrices (list of np.array): Precomputed FST matrices.

    Returns:
        None. Shows matplotlib plots.
    """
    n = len(fst_matrices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]  # Ensure axes is iterable for a single matrix

    for k, matrix in enumerate(fst_matrices):
        ax = axes[k]
        im = ax.imshow(matrix, cmap='coolwarm')
        for i in range(len(populations)):
            for j in range(len(populations)):
                ax.text(j, i, round(matrix[i, j], 2),
                        ha="center", va="center")  # Overlay text on heatmap
        ax.set_xticks(range(len(populations)), labels=[f"pop {i + 1}" for i in range(len(populations))])
        ax.set_yticks(range(len(populations)), labels=[f"pop {i + 1}" for i in range(len(populations))])
        ax.set_title(f'{(generations - fst_generations[k]) * generation_step} years ago')

    plt.tight_layout()
    plt.show()


def calculate_SFS(populations):
    """
    Calculates the folded Site Frequency Spectrum (SFS) for each population.

    Parameters:
        populations (list): List of population dictionaries.

    Returns:
        list of np.array: One SFS vector per population.
    """
    sfs = []
    for j, population in enumerate(populations):
        sfs.append(np.zeros(population[CUR_SIZE] + 1))  # Frequency bins
        freqs = population[CUR_FREQS]
        for i in range(num_of_variants):
            f_rare = min(freqs[i], 1 - freqs[i])  # Folded frequency
            count = int(f_rare * population[CUR_SIZE] * 2)
            sfs[j][count] += 1
    return sfs


def show_sfs(sfs_list, num_bins=100):
    """
    Displays a binned heatmap of the folded SFS for all populations.

    Parameters:
        sfs_list (list of np.array): Folded SFS per population.
        num_bins (int): Number of bins to aggregate SFS values into.

    Returns:
        None. Shows a heatmap.
    """
    num_pops = len(sfs_list)
    max_count = max(len(sfs) for sfs in sfs_list)

    heatmap_matrix = np.zeros((num_pops, num_bins))
    bins_edges = np.linspace(0, max_count - 1, num_bins + 1)

    for i, sfs in enumerate(sfs_list):
        bin_indices = np.digitize(np.arange(len(sfs)), bins_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        for idx, count in enumerate(sfs):
            heatmap_matrix[i, bin_indices[idx]] += count

    plt.figure(figsize=(12, 5))
    im = plt.imshow(heatmap_matrix, aspect='auto', cmap='coolwarm', origin='lower')
    plt.colorbar(im, label='Number of loci')
    plt.xlabel('Frequency')
    plt.ylabel('Population')
    plt.yticks(np.arange(num_pops), [f'Pop {i + 1}' for i in range(num_pops)])
    plt.title(f'Folded SFS Heatmap (binned into {num_bins} bins)')
    plt.show()


def selective_alleles_check(populations):
    """
    Identifies selective alleles that are nearly fixed (low variability).

    Parameters:
        populations (list): List of population dictionaries.

    Returns:
        list: Indices of variants considered selective (near fixation).
    """
    selective_alleles = []
    for i in range(num_of_variants):
        variant_avg_freq = 0
        for pop in populations:
            variant_avg_freq += pop[CUR_FREQS][i]
        variant_avg_freq /= len(populations)
        # Check near-fixation threshold
        if variant_avg_freq < frequency_threshold or variant_avg_freq > 1 - frequency_threshold:
            selective_alleles.append(i)
    return selective_alleles


def plot_heterozygosity_over_time(populations):
    """
    Plots the average heterozygosity over time for each population.

    Parameters:
        populations (list): List of population dictionaries.

    Returns:
        None. Displays matplotlib line plot.
    """
    plt.figure(figsize=(12, 6))
    for i, population in enumerate(populations):
        heterozygosity = []
        freqs_over_time = population[FREQS]
        start_gen = population[START_GEN]
        gens = list(range(start_gen, start_gen + len(freqs_over_time)))

        for freqs in freqs_over_time:
            H_per_variant = [2 * p * (1 - p) for p in freqs]
            avg_H = np.mean(H_per_variant)
            heterozygosity.append(avg_H)

        plt.plot(gens, heterozygosity, label=f'Population {i + 1} - {populations[i][AREA]}')

    plt.xlabel('Generation')
    plt.ylabel('Average Heterozygosity')
    plt.title('Average Heterozygosity Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()