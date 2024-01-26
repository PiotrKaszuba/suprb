import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from plan_on_command.examples.cartpole_example import RandomDataGenerator
from suprb import SupRB, rule
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.rule import origin
from suprb.optimizer.rule.mutation import HalfnormIncrease
from suprb.optimizer.rule.ns import ns
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyFitnessBiased
from suprb.optimizer.solution import ga



if __name__ == '__main__':
    random_state = 42

    # Prepare the data
    random_data_gen = RandomDataGenerator(
        select_X_dimensions=
        [0, 1, 2, 3],
        select_y_dimensions= 2,
        # [0, 1, 2, 3],
    )

    X, y, a = random_data_gen.get_X_y_a_random_data(num_experience_samples=100)

    X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(X, y, a, test_size=0.2, random_state=random_state)

    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape((-1, 1))).reshape((-1,))
    y_test = y_scaler.transform(y_test.reshape((-1, 1))).reshape((-1,))

    all_a_train = np.unique(a_train)

    # Prepare the model
    models = {a: SupRB(
        rule_generation=ns.NoveltySearch(
            init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                              model=Ridge(alpha=0.01,
                                                          random_state=random_state)),
            origin_generation=origin.SquaredError(),
            mutation=HalfnormIncrease(),
            novelty_calculation=NoveltyFitnessBiased()
        ),
        solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32),
        n_iter=32,
        n_rules=8,
        verbose=10,
        logger=CombinedLogger(
            [('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )
        for a in all_a_train}

    for a in all_a_train:
        models[a].fit(X_train[a_train == a], y_train[a_train == a])

    # Evaluate the model
    y_pred_test = np.zeros_like(y_test)
    y_pred_train = np.zeros_like(y_train)
    for a in all_a_train:
        y_pred_test[a_test == a] = models[a].predict(X_test[a_test == a])
        y_pred_train[a_train == a] = models[a].predict(X_train[a_train == a])

    mse_test = mean_squared_error(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)

    print("y_pred_test")
    print(y_pred_test)

    print("y_pred_train")
    print(y_pred_train)

    print(f"MSE (train): {mse_train}")
    print(f"MSE (test): {mse_test}")



