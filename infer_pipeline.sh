#!/bin/bash

## Partition the Data - No Dependencies
python3 infer_features/partition_data.py
## Train the feature models
python3 infer_features/train_feature_model.py train
## Save the inferred features
while read a b c; do
	python3 infer_features/train_feature_model.py "${a}" "${b}" "${c}"
done < <(sed 's/,/\t/g' infer_features/inputs.csv)