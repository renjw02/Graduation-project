set -e

# # train text2pql-t5-large model
# python -u text2pql.py \
#     --batch_size 8 \
#     --gradient_descent_step 6 \
#     --device "3" \
#     --learning_rate 1e-4 \
#     --epochs 32 \
#     --seed 42 \
#     --save_path "./models/text2pql/advising/base" \
#     --tensorboard_save_path "./tensorboard_log/text2pql-t5-base" \
#     --model_name_or_path "./t5-base" \
#     --use_adafactor \
#     --mode train \
#     --train_filepath "./data/advising/advising_train_t5.json"

# # select the best text2pql-t5-large ckpt
# python -u evaluate_text2pql.py \
#     --batch_size 8 \
#     --device "3" \
#     --seed 42 \
#     --save_path "./models/text2pql/advising/base" \
#     --eval_results_path "./eval_results/text2pql/advising/t5" \
#     --mode eval \
#     --dev_filepath "./data/advising/advising_dev_t5.json" \
#     --db_path "./data/advising/advising-db.added-in-2020.sqlite" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "advising" 

# restaurants 
# train text2pql-t5-large model
python -u text2pql.py \
    --batch_size 8 \
    --gradient_descent_step 6 \
    --device "1" \
    --learning_rate 1e-4 \
    --epochs 64 \
    --seed 42 \
    --save_path "./models/text2pql/restaurants/base/split_1" \
    --tensorboard_save_path "./tensorboard_log/text2pql-t5-base" \
    --model_name_or_path "./t5-base" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/restaurants/cross_validation/split_1/train.json"

# select the best text2pql-t5-large ckpt
python -u evaluate_text2pql.py \
    --batch_size 8 \
    --device "1" \
    --seed 42 \
    --save_path "./models/text2pql/restaurants/base/split_1" \
    --eval_results_path "./eval_results/text2pql/restaurants/base" \
    --mode eval \
    --dev_filepath "./data/restaurants/cross_validation/split_1/test.json" \
    --db_path "./data/restaurants/restaurants-db.added-in-2020.sqlite" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --dataset "restaurants" 

# imdb
# train text2pql-t5-large model
# python -u text2pql.py \
#     --batch_size 8 \
#     --gradient_descent_step 6 \
#     --device "3" \
#     --learning_rate 1e-4 \
#     --epochs 208 \
#     --seed 42 \
#     --save_path "./models/text2pql/imdb/base/split_2" \
#     --tensorboard_save_path "./tensorboard_log/text2pql-t5-base" \
#     --model_name_or_path "./t5-base" \
#     --use_adafactor \
#     --mode train \
#     --train_filepath "./data/imdb/cross_validation_t5/split_2/train.json"

# # select the best text2pql-t5-large ckpt
# python -u evaluate_text2pql.py \
#     --batch_size 8 \
#     --device "3" \
#     --seed 42 \
#     --save_path "./models/text2pql/imdb/base/split_2" \
#     --eval_results_path "./eval_results/text2pql/imdb/split_2" \
#     --mode eval \
#     --dev_filepath "./data/imdb/cross_validation_t5/split_2/test.json" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "imdb" 

# # wikisql
# python -u text2pql.py \
#     --batch_size 8 \
#     --gradient_descent_step 6 \
#     --device "1" \
#     --learning_rate 1e-4 \
#     --epochs 64 \
#     --seed 42 \
#     --save_path "./models/text2pql/wikisql/t5" \
#     --tensorboard_save_path "./tensorboard_log/text2pql-t5-base" \
#     --model_name_or_path "./t5-base" \
#     --use_adafactor \
#     --mode train \
#     --train_filepath "./data/wikisql/t5/wikisql_train_preprocessed.json"

# # select the best text2pql-t5-large ckpt
# python -u evaluate_text2pql.py \
#     --batch_size 8 \
#     --device "1" \
#     --seed 42 \
#     --save_path "./models/text2pql/wikisql/t5" \
#     --eval_results_path "./eval_results/text2pql/wikisql/t5" \
#     --mode eval \
#     --dev_filepath "./data/wikisql/t5/wikisql_dev_preprocessed.json" \
#     --db_path "./data/wikisql/dev.db" \
#     --table_file_path "./data/wikisql/dev.tables.jsonl" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "wikisql" 
