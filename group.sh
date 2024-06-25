#!/bin/bash

# Define the list of files (without directory paths)
files=("DC_Hmi" "DOA_Hmi" "DT_Hmi" "NnA_Hmi" "NnT_Hmi")

# Input and output directories
input_dir="./WER/Meta/HMI"
output_dir="./ASR/io"
output_file="speaker_to_group.txt"

# Temporary file to store intermediate results
temp_file=$(mktemp)

# Process each file
for file in "${files[@]}"; do
    # Extract the prefix before the underscore in the filename
    file_prefix=$(echo "$file" | cut -d'_' -f1)

    # Process the file and append results to the temporary file
    cut -d '-' -f 1 "$input_dir/$file" | sort | uniq | sed 's/^./n/' | while read -r line; do
        echo "$line $file_prefix"
    done >> "$temp_file"
done

# Ensure the output directory exists
mkdir -p "$output_dir"

# Sort the final output and redirect it to the specified output file
sort "$temp_file" > "$output_dir/$output_file"

# Clean up the temporary file
rm "$temp_file"

