set -e

# python -u text2pql.py \
#     --batch_size 8 \
#     --gradient_descent_step 6 \
#     --device "1" \
#     --learning_rate 5e-5 \
#     --epochs 128 \
#     --seed 42 \
#     --save_path "./models/restaurants/split_0" \
#     --model_name_or_path "./bert-base-uncased" \
#     --use_adafactor \
#     --mode train \
#     --train_filepath "data/advising/advising_train_t5.json"

# python -u evaluate_text2pql.py \
#     --batch_size 8 \
#     --device "1" \
#     --seed 42 \
#     --save_path "./models/restaurants/split_0" \
#     --eval_results_path "./eval_results/restaurants/split_0" \
#     --mode eval \
#     --dev_filepath "data/advising/advising_dev_t5.json" \
#     --db_path "./data/restaurants/restaurants-db.added-in-2020.sqlite" \

# imdb
# python -u text2pql.py \
#     --batch_size 8 \
#     --gradient_descent_step 6 \
#     --device "2" \
#     --learning_rate 1e-4 \
#     --epochs 64 \
#     --seed 42 \
#     --save_path "./models/imdb/split_4" \
#     --model_name_or_path "./bert-base-cased" \
#     --use_adafactor \
#     --mode train \
#     --train_filepath "data/imdb/cross_validation_t5/split_4/train.json"

# python -u evaluate_text2pql.py \
#     --batch_size 8 \
#     --device "2" \
#     --seed 42 \
#     --save_path "./models/imdb/split_4" \
#     --eval_results_path "./eval_results/imdb/split_4" \
#     --mode eval \
#     --dev_filepath "data/imdb/cross_validation_t5/split_4/test.json" \
#     --db_path "./data/restaurants/restaurants-db.added-in-2020.sqlite" \

# wikisql
# python -u text2pql.py \
#     --batch_size 8 \
#     --gradient_descent_step 6 \
#     --device "3" \
#     --learning_rate 1e-4 \
#     --epochs 64 \
#     --seed 42 \
#     --save_path "./models/wikisql" \
#     --model_name_or_path "./bert-base-cased" \
#     --use_adafactor \
#     --mode train \
#     --train_filepath "data/wikisql/t5/wikisql_train_preprocessed.json"

python -u evaluate_text2pql.py \
    --batch_size 8 \
    --device "3" \
    --seed 42 \
    --save_path "./models/wikisql" \
    --eval_results_path "./eval_results/wikisql" \
    --mode eval \
    --dev_filepath "data/wikisql/t5/wikisql_dev_preprocessed.json" \
    --db_path "./data/restaurants/restaurants-db.added-in-2020.sqlite" 