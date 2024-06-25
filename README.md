# Exploring the Relationship Between Bias and Speech Acoustics in Automatic Speech Recognition Systems

### General
This repository contains the code used for my Bachelor thesis conducted for [Research Project](https://github.com/TU-Delft-CSE/Research-Project) 2024 at the [TU Delft](https://github.com/TU-Delft-CSE). The topic of the project was "Exploring the Relationship Between Bias and Speech Acoustics in Automatic
Speech Recognition Systems".

### File structure
- `fetch.py`: fetches the speech files from the dataset
- `group.sh`: gathers the metadata about the speaker groups
- `calculate_bias.sh`: calculates the bias based on the word error rates
- `featurize.py`: computes the acoustic embeddings for speech files
- `distance.py`: computes the distance between the acoustic embeddings
- `plot.py`: plots scatter plots for bias against the acoustic distance
- `requirements.txt`: lists the libraries and their versions used for the project

The scripts assume the existence of data that is not present in this repository for licensing reasons. The code should be adapted to the dataset structure.
