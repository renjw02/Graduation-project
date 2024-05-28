set -e

# train text2pql-t5-large model
# python -u text2pql.py \
#     --batch_size 4 \
#     --gradient_descent_step 6 \
#     --device "2" \
#     --learning_rate 5e-5 \
#     --epochs 64 \
#     --seed 42 \
#     --save_path "./models/bart-large/advising" \
#     --tensorboard_save_path "./tensorboard_log/text2pql-t5-large" \
#     --model_name_or_path "./bart-large" \
#     --use_adafactor \
#     --mode train \
#     --train_filepath "data/advising/advising_train_t5.json"

# # select the best text2pql-t5-large ckpt
# python -u evaluate_text2pql.py \
#     --batch_size 4 \
#     --device "2" \
#     --seed 42 \
#     --save_path "./models/bart-large/advising" \
#     --eval_results_path "./eval_results/bart-large/advising" \
#     --mode eval \
#     --dev_filepath "data/advising/advising_dev_t5.json" \
#     --db_path "./data/advising/advising-db.added-in-2020.sqlite" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "advising" 

# restaurants 
# train text2pql-t5-large model
# python -u text2pql.py \
#     --batch_size 4 \
#     --gradient_descent_step 6 \
#     --device "3" \
#     --learning_rate 5e-5 \
#     --epochs 64 \
#     --seed 42 \
#     --save_path "./models/bart-large/restaurants/split_4" \
#     --tensorboard_save_path "./tensorboard_log/text2pql-t5-large" \
#     --model_name_or_path "./bart-large" \
#     --use_adafactor \
#     --mode train \
#     --train_filepath "./data/restaurants/t5/split_4/train.json"

# # select the best text2pql-t5-large ckpt
# python -u evaluate_text2pql.py \
#     --batch_size 4 \
#     --device "3" \
#     --seed 42 \
#     --save_path "./models/bart-large/restaurants/split_4" \
#     --eval_results_path "./eval_results/text2pql/restaurants/split_4" \
#     --mode eval \
#     --dev_filepath "./data/restaurants/t5/split_4/test.json" \
#     --db_path "./data/restaurants/restaurants-db.added-in-2020.sqlite" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "restaurants" 

# imdb
# train text2pql-t5-large model
# python -u text2pql.py \
#     --batch_size 2 \
#     --gradient_descent_step 6 \
#     --device "3" \
#     --learning_rate 5e-5 \
#     --epochs 128 \
#     --seed 42 \
#     --save_path "./models/text2pql/imdb/split_3" \
#     --tensorboard_save_path "./tensorboard_log/text2pql-t5-large" \
#     --model_name_or_path "./t5-large" \
#     --use_adafactor \
#     --mode train \
#     --train_filepath "./data/imdb/cross_validation/split_3/train.json"

# # select the best text2pql-t5-large ckpt
# python -u evaluate_text2pql.py \
#     --batch_size 2 \
#     --device "3" \
#     --seed 42 \
#     --save_path "./models/text2pql/imdb/split_3" \
#     --eval_results_path "./eval_results/text2pql/imdb/split_3" \
#     --mode eval \
#     --dev_filepath "./data/imdb/cross_validation/split_3/test.json" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "imdb" 

# wikisql
# python -u text2pql.py \
    # --batch_size 4 \
    # --gradient_descent_step 6 \
    # --device "1" \
    # --learning_rate 5e-5 \
    # --epochs 64 \
    # --seed 42 \
    # --save_path "./models/bart-large/wikisql" \
    # --tensorboard_save_path "./tensorboard_log/text2pql-t5-large" \
    # --model_name_or_path "./bart-large" \
    # --use_adafactor \
    # --mode train \
    # --train_filepath "data/wikisql/t5/wikisql_train_preprocessed.json"

# select the best text2pql-t5-large ckpt
python -u evaluate_text2pql.py \
    --batch_size 4 \
    --device "1" \
    --seed 42 \
    --save_path "./models/bart-large/wikisql" \
    --eval_results_path "./eval_results/bart-large/wikisql" \
    --mode eval \
    --dev_filepath "data/wikisql/t5/wikisql_dev_preprocessed.json" \
    --db_path "./data/wikisql/dev.db" \
    --table_file_path "./data/wikisql/dev.tables.jsonl" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --dataset "wikisql" 