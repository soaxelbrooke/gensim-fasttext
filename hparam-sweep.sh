
mkdir -p out
for EPOCHS in 5 50 100; do
    for WINDOW_LEN in 3 5 7; do
        for NEG_SAMP_DIST in -0.5 0.75; do
            export OUTDIR="out/e${EPOCHS}_wl${WINDOW_LEN}_nsd${NEG_SAMP_DIST}"
            echo "Training vectors with epochs=$EPOCHS, window length=$WINDOW_LEN, neg sampling exponent=$NEG_SAMP_DIST..."
            time python3 train.py train
	    mkdir -p $OUTDIR
            mv vectors_out.sqlite $OUTDIR/vectors.sqlite
        done
    done
done

