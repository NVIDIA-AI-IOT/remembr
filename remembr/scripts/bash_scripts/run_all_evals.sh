for seed in 0 1 2;
    for i in 0 3 4 6 16 21 22
    do

        python scripts/eval.py --model remembr+command-r --sequence_id $i --num_ctx 65536 --postfix "_$seed" -caption_file captions_VILA1.5-13b_3_secs
        python scripts/eval.py --model remembr+gpt-4o --sequence_id $i --postfix "_$seed" --caption_file captions_VILA1.5-13b_3_secs

    done
done
