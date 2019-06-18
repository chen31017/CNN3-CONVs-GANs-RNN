cd .. #git will sync this into hw3, but it needs to run out of assignment3 folder
pwd

for k in {32,64}
do
    for L in {2,4,8,16}
    do
        echo "==============================================="
        echo "running ${L} layers with ${k} filters per layer"
        echo "==============================================="
        python -m hw3.experiments run-exp \
        --run-name exp1_1_K${k}_L${L} --epochs 100 --early-stopping 2 --lr 1e-3\
        --filters-per-layer ${k} --layers-per-block ${L} --pool-every 2 --hidden-dims 1024;
    done;
done;