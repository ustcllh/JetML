from hyperopt import fmin, tpe, hp


def obj(args):
    print(args)
    return 0


space = hp.choice('hyper parameters', [
    {
        'num_batch': hp.choice('num_batch', [100, 200, 300, 400]),
        'num_layers': hp.quniform('num_layers', 5, 10, 1),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.01),
        'decay_factor': hp.uniform('decay_factor', 0.1, 0.8),
        'num_epochs':hp.quniform('num_epochs', 5, 30, 5)
    }
])

best = fmin(obj, space, algo=tpe.suggest, max_evals=10)

print(best)
