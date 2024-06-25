import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def main():
    print("Starting plotting", flush=True)
    root_directory = "/scratch/ppcichon/ASR/io/RD/results_full/"
    group_min_abs_correlation_coefficients = []
    group_min_rel_correlation_coefficients = []

    for asr_model in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, asr_model)
        if os.path.isdir(subdir_path):
            group_min_abs_biases = {}
            group_min_rel_biases = {}
            bias_file_path = os.path.join(subdir_path, 'speaker_to_bias.txt')
            if os.path.isfile(bias_file_path):
                with open(bias_file_path, 'r') as file:
                    for line in file:
                        columns = line.strip().split()
                        if len(columns) >= 4:
                            key = columns[0]
                            value_3 = columns[2]
                            value_4 = columns[3]
                            group_min_abs_biases[key] = value_3
                            group_min_rel_biases[key] = value_4
            for embedding_model in os.listdir(subdir_path):
                subdir_path = os.path.join(root_directory, asr_model, embedding_model)
                if os.path.isdir(subdir_path):
                    group_min_abs_distances = {}
                    group_min_rel_distances = {}
                    bias_file_path = os.path.join(subdir_path, 'speaker_to_distance.txt')
                    if os.path.isfile(bias_file_path):
                        with open(bias_file_path, 'r') as file:
                            for line in file:
                                columns = line.strip().split()
                                if len(columns) >= 3:
                                    key = columns[0]
                                    value_2 = columns[1]
                                    value_3 = columns[2]
                                    group_min_abs_distances[key] = value_2
                                    group_min_rel_distances[key] = value_3
                    corr, p = plot(group_min_abs_biases, group_min_abs_distances,
                                   os.path.join(subdir_path, "scatter_test_run.png"))
                    group_min_abs_correlation_coefficients.append((asr_model, embedding_model, (corr, p)))
                    # corr, _ = plot(group_min_rel_biases, group_min_rel_distances,
                    #                os.path.join(subdir_path, "scatter-rel.png"))
                    # group_min_rel_correlation_coefficients.append((asr_model, embedding_model, corr))
    print(group_min_abs_correlation_coefficients)
    # plot_table(group_min_abs_correlation_coefficients, os.path.join(root_directory, "correlation-abs.png"))
    # plot_table(group_min_rel_correlation_coefficients, os.path.join(root_directory, "correlation-rel.png"))


def plot_table(data, output_path):
    # Extracting data for plotting
    labels = [entry[0] for entry in data]
    values = [entry[2] for entry in data]

    # Convert data to lists for table creation
    labels = [entry[0] for entry in data]
    categories = [entry[1] for entry in data]
    values = [entry[2] for entry in data]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Hide axes
    ax.axis('off')

    # Create the table
    table_data = []
    for label, category, value in data:
        table_data.append([label, category, value])

    # Table formatting
    table = ax.table(cellText=table_data, colLabels=['Label', 'Category', 'Value'], loc='center')

    # Table styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title('Table with String Labels and Float Values', pad=20)  # Add title to the table

    plt.savefig(output_path)
    plt.close()
    print("Saved correlation table")


def plot(xs, ys, output_path):
    matching_keys = set(xs.keys()).intersection(ys.keys())
    biases = [round(float(xs[key]), 4) for key in matching_keys]
    distances = [round(float(ys[key]), 4) for key in matching_keys]
    plt.figure()
    plt.scatter(biases, distances)
    plt.xlabel('Bias')
    plt.ylabel('Acoustic distance')
    plt.savefig(output_path)
    plt.close()
    print("Saved " + output_path)
    print("Biases ", len(biases))
    print("Distances", len(distances))
    corr, p = pearsonr(biases, distances)
    print(corr)
    return corr, p


if __name__ == '__main__':
    main()

