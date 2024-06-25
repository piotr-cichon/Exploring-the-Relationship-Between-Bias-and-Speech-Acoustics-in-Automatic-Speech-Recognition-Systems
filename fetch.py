import os
import shutil

print("Starting the fetching of speech files")

# segment_metadata_directory = "./ASR/WER/Meta/Rd/"
segment_metadata_directory = "./ASR/WER/Meta/HMI/"

files = os.listdir(segment_metadata_directory)

speakers = []
for file_name in files:
    file_path = os.path.join(segment_metadata_directory, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            print("Opening " + file_name)
            for line in file:
                index = line.find(' ')
                if index != -1:
                    substring = line[:index]
                else:
                    substring = line.strip()
                split = substring.split("-")
                if len(split) < 3:
                    continue
                speaker_id = split[0].replace("N", "n")
                if speaker_id in speakers:
                    continue
                speakers.append(speaker_id)
                print(speaker_id)
                utterance_id = split[1]
                path = "./ASR/JASMIN/Data/data/audio/wav/comp-p/nl/" + utterance_id + ".wav"
                if not os.path.isfile(path):
                    print("Failed to find " + utterance_id)
                    continue
                audio_file_directory = "./ASR/io/HMI/recordings_full/" + speaker_id
                if not os.path.exists(audio_file_directory):
                    os.makedirs(audio_file_directory)
                shutil.copy(path, audio_file_directory + "/" + speaker_id + ".wav")
                print("Exported " + utterance_id)
