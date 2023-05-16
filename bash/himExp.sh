ROOT=.

python $ROOT/exp.py \
--datadir $ROOT/data/sponge/test \
--savedir $ROOT/result/himExp \
--solverdir $ROOT/data/solvers/himSolvers.csv \
--etaRange 2. 8. \
--lamb 20 \
--n 200 \
--maxIter 100 \
--modeldir $ROOT/model/std25.ckpt \
--device cpu