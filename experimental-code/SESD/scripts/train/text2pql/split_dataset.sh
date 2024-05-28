set -e 

python -u cross_validation.py \
    --data_path "data/restaurants/restaurants_preprocessed.json" \
    --output_path "data/restaurants/cross_validation" \
    --n_splits 5 \
    --seed 42