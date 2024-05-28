set -e

# # test the best text2sql-t5-large ckpt
# python -u accuracy.py \
#     --batch_size 8 \
#     --device "3" \
#     --seed 42 \
#     --save_path "./models/text2pql/advising/base/checkpoint-19356" \
#     --dev_filepath "./data/advising/advising_test_t5.json" \
#     --mode eval \
#     --db_path "./data/advising/advising-db.added-in-2020.sqlite" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "advising" \
#     --output "predictions/advising/pred.sql"


# python -u accuracy.py \
#     --batch_size 2 \
#     --device "2" \
#     --seed 42 \
#     --save_path "./models/text2pql/restaurants/split_0/checkpoint-10192" \
#     --dev_filepath "./data/restaurants/cross_validation/split_1/test.json" \
#     --mode eval \
#     --db_path "./data/restaurants/restaurants-db.added-in-2020.sqlite" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "restaurants" \
#     --output "predictions/restaurants/pred.sql"

# python -u accuracy.py \
#     --batch_size 8 \
#     --device "1" \
#     --seed 42 \
#     --save_path "./best_model/imdb/base/checkpoint-1300" \
#     --dev_filepath "./data/imdb/cross_validation/split_0/test.json" \
#     --mode eval \
#     --db_path "./data/restaurants/restaurants-db.added-in-2020.sqlite" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "imdb" \
#     --table_file_path "./data/imdb/table_imdb.json" \
#     --output "predictions/imdb/pred.sql"

# python -u accuracy.py \
#     --batch_size 8 \
#     --device "3" \
#     --seed 42 \
#     --save_path "./models/text2pql/wikisql/decoder/checkpoint-100632" \
#     --dev_filepath "./data/wikisql/wikisql_test_preprocessed.json" \
#     --mode eval \
#     --db_path "./data/wikisql/test.db" \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --dataset "wikisql" \
#     --table_file_path "data/wikisql/test.tables.jsonl" \
#     --output "predictions/wikisql/pred.sql"

python -u accuracy.py \
    --batch_size 8 \
    --device "2" \
    --seed 42 \
    --save_path "./models/bart-large/wikisql/checkpoint-100633" \
    --dev_filepath "./data/wikisql/wikisql_test_preprocessed.json" \
    --mode eval \
    --db_path "./data/wikisql/test.db" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --dataset "wikisql" \
    --table_file_path "data/wikisql/test.tables.jsonl" \
    --output "predictions/wikisql/pred.sql"