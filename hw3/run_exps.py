from experiments.py import run_experiment

def main():
    s = 'exp1_1_K{}_L{}'
    for L in {2,4,8,16}:
        run_experiment(s.format(32,L), out_dir='./results', seed=None,
                       # Training params
                       bs_train=128*2, bs_test=128*2, batches=100, epochs=100,
                       early_stopping=3, checkpoints=None, lr=1e-2, reg=2e-3,
                       # Model params
                       filters_per_layer=[32], layers_per_block=L, pool_every=2,
                       hidden_dims=[1024], ycn=False)

if __name__ == '__main__':
    main()