cd .. #git will sync this into hw3, but it needs to run out of assignment3 folder
pwd

for L in {1,2,3,4}
do
        echo "==============================================="
        echo "running ${L} layers with ${k} filters per layer"
        echo "==============================================="
        python -m hw3.experiments run-exp \
        --run-name exp1_2_L${L}_K64-128-256 --epochs 100 --early-stopping 2 --lr 1e-3 \
        --filters-per-layer 64 128 256 --layers-per-block ${L} --pool-every 2 --hidden-dims 1024;
done;