To evaluate on NaVQA, we provide the following instructions for downloading, preprocessing, and evaluating on the data.

## Download and preprocess the CODa dataset
First download the relevant subsets of the [CODa dataset](https://amrl.cs.utexas.edu/coda/), which consists of 22 sequences. 

We only need 7 of them which are `0, 3, 4, 6, 16, 21, 22`. These numbers will be referred to as sequence IDs. Each sequence ID has 30 questions associated with it.

> Because of the number of videos, be sure to have a large amount of storage. The processed dataset is ~335GB, but since the pre-processing phase also downloads LiDAR and other outputs, we would recommend having ~500GB extra storage.

Download the CODa devkit to some directory not inside ReMEmbR:
```
git clone https://github.com/ut-amrl/coda-devkit.git
```

Then let us set a few environment variables. Fill them with the appropriate paths. The `REMEMBR_PATH` is the folder where the `scripts` folder is accessible.
We would recommend adding these to your `~/.bashrc`
```
export CODA_ROOT_DIR=/path/to/coda-devkit/data
export REMEMBR_PATH=/path/to/remembr
cd $CODA_ROOT_DIR/..
```

Then run the following command which will preprocess the data in the appropriate format:

```
cd remembr
bash scripts/bash_scripts/preprocess_coda_all.sh
```

## Caption the dataset offline

Ensure the location of your preprocessed coda data is located in `/path/to/remembr/coda_data`

Given the dataset, run the following command for each. We describe the meaning of each below:


```
python scripts/preprocess_captions.py \
    --seq_id 0 \
    --seconds_per_caption 3 \
    --model-path Efficient-Large-Model/VILA1.5-13b
    --captioner_name VILA1.5-13b 
    --out_path data/captions/0/captions
```

- `seq_id`: The sequence ID from the CODa dataset (of the 7 listed in the previous section)
- `seconds_per_caption`: The number of seconds of frames aggregated for generating a caption
- `model-path`: The name of the specific VILA model as described in their [code](https://github.com/NVlabs/VILA/tree/main)
- `captioner_name`: The name of the output file prefix based on the captioner type
- `out_path`: The format of the captions must be: `data/captions/{seq_id}/captions`

Be sure to set the `captioner_name` correctly so that it matches the model used in `model-path`!

The captions for each frame should be put into a JSON file located in `data/captions/{seq_id}/captions`.

We provide an example to preprocess all captions as above in `scripts/bash_scripts/preprocess_captions_all.sh`

## Download the dataset and preprocess it

### 1. Download `human_unfilled` into the `data` folder.

```
TODO. ADD DATASET DOWNLOAD INSTRUCTIONS
```

This contains templates of the questions as json files for each sequence. Then, there is a `data.csv` that includes human annotated questions. 


### 2. Form the questions in the proper format
Run the following script, providing it a base captioner file that you ran previously. 

```
python scripts/question_scripts/form_question_jsons.py --caption_file captions_{{captioner_name}}_{{seconds_per_caption}}_secs
```

This is meant to also aggregate the "optimal" context required to answer the question based on the captioner and seconds per caption, so you must set `captioner_name` and `seconds_per_caption`. We recommend using a 3 seconds per caption value. Here is an example:

```
python scripts/question_scripts/form_question_jsons.py --caption_file captions_VILA1.5-13b_3_secs
```

After this step, a folder called `data/questions` should exist.


## Run the evaluation

To run the evaluation, you must first run the MilvusDB container. All evaluations create a MilvusDB collection per sequence ID.
```
python scripts/eval.py \
    --model {{eval_method}} \
    --sequence_id {{seq_id}} \
    --caption_file captions_{{captioner_name}}_{{seconds_per_caption}}_secs \
    --postfix {{postfix}} 
```

Because of how the code is written, if `seconds_per_caption` is changed, we would recommend re-running `questions/form_question_jsons.py` 

To continue the example on sequence ID 0, we show an example here:
```
python scripts/eval.py \
    --model remembr+llama3.1:8b \
    --sequence_id 0 \
    --caption_file captions_VILA1.5-13b_3_secs \
    --postfix _0
```

An example of running `eval.py` across multiple tries and across all sequences, look at `scripts/bash_scripts/run_all_evals.sh`