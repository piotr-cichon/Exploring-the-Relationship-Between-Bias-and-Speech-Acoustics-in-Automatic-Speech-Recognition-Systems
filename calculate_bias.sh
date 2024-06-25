#!/bin/bash

directory="BIAS"

find "$directory" -mindepth 2 -type f -name "speaker_to_wer.txt" | while read -r file; do
    dir=$(dirname "$file")
    X=$(basename "$dir")

    min_Y=$(awk '{print $2}' "$file" | sort -n | head -n 1)

    echo "speaker WER group_to_min_abs group_to_min_rel" > "$dir/speaker_to_bias.txt"

    while IFS=' ' read -r X Y; do
        # Calculate new values
        group_to_min_abs=$(echo "$Y - $min_Y" | bc)
        group_to_min_rel=$(echo "($Y - $min_Y) / $min_Y" | bc -l)

        # Append Y Z group_to_min_abs group_to_min_rel to the temporary file
        echo "$X $Y $group_to_min_abs $group_to_min_rel" >> "$dir/speaker_to_bias.txt"
    done < "$file"
done

