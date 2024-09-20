
for i in 0 3 4 6 16 21 22;
do

    python scripts/preprocess_captions.py --seq_id $i --seconds_per_caption 3 --captioner_name VILA1.5-13b --model-path Efficient-Large-Model/VILA1.5-13b --out_path data/captions/$i/captions

done




