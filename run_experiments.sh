WORKERS=${1:-4}
SEEDS=${2:-10}
parallel -j ${WORKERS} "python3 src/OLR_main.py -algo CP -D {1} -s {2}" ::: $(seq 0.5 0.5 3.5) ::: $(seq 1 1 $SEEDS)
parallel -j ${WORKERS} "python3 src/OLR_main.py -algo CS -D {1} -s {2}" ::: $(seq 0.5 0.5 3.5) ::: $(seq 1 1 $SEEDS)
parallel -j ${WORKERS} "python3 src/OLR_main.py -algo CRDG -D {1} -s {2}" ::: $(seq 0.5 0.5 3.5) ::: $(seq 1 1 $SEEDS)
parallel -j ${WORKERS} "python3 src/OLR_main.py -algo ADAGRAD -s {1}" ::: $(seq 1 1 $SEEDS)
parallel -j ${WORKERS} "python3 src/OLR_main.py -algo OGD -s {1}" ::: $(seq 1 1 $SEEDS)
python3 src/IMDB_main.py
python3 src/SPAM_main.py
