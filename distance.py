# The code was inspired, with part of the code used from the code associated with the paper: Quantifying Language Variation Acoustically with Few Resources.
# @misc{bartelds-2022-quantifying,
#  title = {{Quantifying Language Variation Acoustically with Few Resources}},
#  author = {Bartelds, Martijn and Wieling, Martijn},
#  year = {2022},
#  publisher = {arXiv},
#  url = {https://arxiv.org/abs/2205.02694},
#  doi = {10.48550/ARXIV.2205.02694}
# }

import os
from dtw import dtw
import numpy as np
from glob import glob
import sys


def main():
    print("Starting calculating the distances", flush=True)

    speech_types = ["RD", "HMI"]
    embedding_models = ['wav2vec2-large-960h', 'wav2vec2-large-nl-ft-cgn', 'wav2vec2-large-xlsr-53-ft-cgn']
    asr_models = ["conformer_noaugs", "conformer_sp_specaug", "whisper-small-test", "conformer_onlysp", "whisper-finetune-small-withoutsp_withoutspecaug"]

    speech_type = speech_types[int(sys.argv[1])]
    embedding_model = embedding_models[int(sys.argv[2])]
    asr_model = asr_models[int(sys.argv[3])]
    num_features = int(sys.argv[4])  # out of 1024
    num_speakers = int(sys.argv[5])

    root_directory = os.path.join("./ASR/io", speech_type, "results_full/")
    for subdir in os.listdir(root_directory):
        print(subdir, asr_model)
        if not subdir == asr_model:
            continue
        subdir_path = os.path.join(root_directory, subdir)
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
            group_min_abs_speaker = min(group_min_abs_biases, key=group_min_abs_biases.get)
            group_min_rel_speaker = min(group_min_rel_biases, key=group_min_rel_biases.get)
            compute_distances(group_min_abs_speaker, group_min_rel_speaker, subdir, embedding_model, speech_type, num_features, num_speakers)


def compute_distances(group_min_abs_speaker, group_min_rel_speaker, asr_model, embedding_model, speech_type, num_features, num_speakers):
    files = glob(os.path.join("./ASR/io", speech_type, "embeddings_full", embedding_model, "*.npy"))
    # files = glob(os.path.join("./test-embeddings7", "*.npy"))[:50]

    embeddings = dict()

    print("Computing the distances for " + asr_model + " model between the " + embedding_model + " embeddings for ", speech_type, " speech", flush=True)
    for feat in sorted(files):
        speaker_name = os.path.splitext(os.path.basename(feat))[0].split("-")[0]
        embedding_path = feat
        embeddings[speaker_name] = embedding_path

    group_min_rel_speaker_features = np.load(embeddings[group_min_rel_speaker])
    group_min_rel_speaker_features = group_min_rel_speaker_features[:, :num_features]

    group_min_rel_distances = dict()
    for speaker_name in list(embeddings)[:num_speakers]:
        print("Current speaker " + speaker_name, flush=True)
        speaker_features = np.load(embeddings[speaker_name])
        speaker_features = speaker_features[:, :num_features]
        print("Numer of features is ", np.shape(speaker_features), np.shape(group_min_rel_speaker_features), flush=True)
        dobj = dtw(group_min_rel_speaker_features, speaker_features)
        current_group_min_rel_distance = dobj.normalizedDistance
        print("Computed distance ", current_group_min_rel_distance)
        group_min_rel_distances[speaker_name] = round(current_group_min_rel_distance, 4)

    output_file_path = "./ASR/io/" + speech_type + "/results_full/" + asr_model + "/" + embedding_model + "/speaker_to_distance_all_1.txt"
    # output_file_path = os.path.normpath("./distances7/" + model + "/speaker_to_distance.txt")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, 'w') as file:
        for (speaker_name, group_min_abs_distance), (_, group_min_rel_distance) in zip(
                group_min_rel_distances.items(), group_min_rel_distances.items()):
            file.write(f'{speaker_name} {group_min_abs_distance} {group_min_rel_distance}\n')


if __name__ == '__main__':
    main()

