for i in 0 3 4 6 16 21 22;
do

    # need to download the data
    cd $CODA_ROOT_DIR/..
    python scripts/download_split.py -d ./data -t sequence -se $i <<< "Y"
    
    # need to process the data
    cd $REMEMBR_PATH
    python scripts/preprocess_coda.py -s $i

    # delete original data
    rm -rf $CODA_ROOT_DIR/*
done




