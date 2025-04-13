#! /bin/bash

for file in /Users/egg/Projects/Stainalyzer/data/raw/SegPath/*_fileinfo.csv; do 
    train_count=$(grep -c "train" "$file")
    test_count=$(grep -c "test" "$file")
    echo "$(basename "$file"),Train: $train_count, Test: $test_count" >> output_counts.csv
done

cat *_fileinfo.csv | grep "train" | wc -l
cat *_fileinfo.csv | grep "test" | wc -l

