ROOT=.

python $ROOT/exp.py \
--datadir $ROOT/data/sponge/test \
--savedir $ROOT/result/semExp \
--solverdir $ROOT/data/solvers/semSolvers.csv \
--etaRange 1. 2. \
--lamb 50 \
--n 500 \
--maxIter 100 \
--modeldir $ROOT/model/std25.ckpt \
--device cpu