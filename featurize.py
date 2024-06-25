# The code was inspired, with part of the code used from the code associated with the paper: Quantifying Language Variation Acoustically with Few Resources.
# @misc{bartelds-2022-quantifying,
#  title = {{Quantifying Language Variation Acoustically with Few Resources}},
#  author = {Bartelds, Martijn and Wieling, Martijn},
#  year = {2022},
#  publisher = {arXiv},
#  url = {https://arxiv.org/abs/2205.02694},
#  doi = {10.48550/ARXIV.2205.02694}
# }

KNOWN_MODELS = {
    # Fine-tuned
    "wav2vec2-large-960h": "facebook/wav2vec2-large-960h",
    "wav2vec2-large-nl-ft-cgn": "GroNLP/wav2vec2-dutch-large-ft-cgn",
    "wav2vec2-large-xlsr-53-ft-cgn": "GroNLP/wav2vec2-large-xlsr-53-ft-cgn"
}

import os
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path


# chooses a random segment of utterance for every speaker
def choose_random_files(speech_type):
    base_directory = os.path.join("./ASR/io", speech_type, "recordings_full")
    random_files = []

    # Iterate over all directories in the base directory
    for root, dirs, files in os.walk(base_directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # List all files in the directory
            file_list = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            # Choose a random file if there are any files
            if file_list:
                random_file = random.choice(file_list)
                full_path = os.path.join(dir_path, random_file)
                random_files.append(full_path)

    return random_files


def load_wav2vec2_featurizer(model, layer=None):
    """
    Loads Wav2Vec2 featurization pipeline and returns it as a function.

    Featurizer returns a list with all hidden layer representations if "layer" argument is None.
    Otherwise, only returns the specified layer representations.
    """
    from transformers.models.wav2vec2 import Wav2Vec2Model
    import soundfile as sf
    import torch
    import numpy as np

    model_name_or_path = KNOWN_MODELS.get(model, model)
    model_kw = {}
    if layer is not None:
        model_kw["num_hidden_layers"] = layer if layer > 0 else 0
    model = Wav2Vec2Model.from_pretrained(model_name_or_path, **model_kw)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    @torch.no_grad()
    def _featurize(path):
        input_values, rate = sf.read(path, dtype=np.float32)
        assert rate == 16_000
        input_values = torch.from_numpy(input_values).unsqueeze(0)
        if torch.cuda.is_available():
            input_values = input_values.cuda()

        # Check if stereo
        if input_values.ndim == 3 and input_values.shape[2] == 2:
            # Convert stereo to mono by averaging the two channels
            input_values = torch.mean(input_values, dim=2)

        hidden_state = model(input_values).last_hidden_state.squeeze(0).cpu().numpy()

        return hidden_state

    return _featurize


def compute_embeddings(model, layer, wav_paths, speech_type):
    output_dir = os.path.join("./ASR/io", speech_type, "embeddings_full")
    # Check wav files in input directory
    # wav_paths = list(input_dir.glob("*.wav"))
    if len(wav_paths) == 0:
        print(f"No wav files found")
        exit(1)
    print(f"Featurizing {len(wav_paths):,} wav files")
    # Load acoustic model
    featurizer = load_wav2vec2_featurizer(model, layer=layer)
    # Create features for each wav file
    for wav_path in tqdm(wav_paths, ncols=80):
        # Check output directory
        speaker = os.path.splitext(wav_path.name)[0]

        output_name = speaker.split('-')[0]
        feat_path = os.path.join(output_dir, model, output_name + ".npy")
        if os.path.isfile(feat_path):
            print("Skipping " + feat_path)
            continue

        # Extract features
        hidden_states = featurizer(wav_path)
        if layer is not None:
            hidden_states = [hidden_states]

        # Save features
        os.makedirs(output_dir, exist_ok=True)
        for l, hidden_state in enumerate(hidden_states, start=layer or 0):
            os.makedirs(os.path.join(output_dir, model), exist_ok=True)
            np.save(feat_path, hidden_state)

        tqdm.write(str(output_dir))
    print("Done! Computed features for " + model + " for type of speech " + speech_type, flush=True)


def main():
    models = ["wav2vec2-large-960h", "wav2vec2-large-nl-ft-cgn", "wav2vec2-large-xlsr-53-ft-cgn"]
    speech_types = ["RD", "HMI"]

    for speech_type in speech_types:
        wav_paths = [Path(path_str) for path_str in choose_random_files(speech_type)]
        compute_embeddings(models[0], 13, wav_paths, speech_type)
        compute_embeddings(models[2], 15, wav_paths, speech_type)
        compute_embeddings(models[1], 16, wav_paths, speech_type)


if __name__ == "__main__":
    main()

