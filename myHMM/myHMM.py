import numpy as np
import datetime
import tushare as ts
import sys

if sys.version_info.major == 3:
    xrange = range


class myHMM(object):
    def __init__(self, tolerance=1e-6, max_iterations=10000):
        self.tolerance = tolerance
        self.max_iter = max_iterations

    def HMMfwd(self, a, b, o, pi):
        N = np.shape(b)[0]
        T = np.shape(o)[0]

        alpha = np.zeros((N, T))
        alpha[:, 0] = pi * b[:, o[0]]

        for t in xrange(1, T):
            for i in xrange(N):
                """
                TODO: Do something to update alpha[i,t]
                """
        return alpha

    def HMMbwd(self, a, b, o):
        # Implements HMM Backward algorithm
        T = np.shape(o)[0]
        N = np.shape(b)[0]
        beta = np.zeros((N, T))
        c = np.ones((T))
        beta[:, T - 1] = c[T - 1]

        for t in xrange(T - 2, -1, -1):
            for i in xrange(N):
                """
                TODO: Do something to update beta[i,t]
                """
        return beta

    def HMMViterbi(self, a, b, o, pi):
        # Implements HMM Viterbi algorithm        

        K = np.shape(b)[0]  # K为隐状态数
        T = np.shape(o)[0]  # T为观测序列长度

        t1 = np.zeros((K, T))  # T_1 in wiki
        t2 = np.zeros((K, T), dtype=np.int64)  # T_2 in wiki

        """
        TODO: implement the viterbi algorithm and return path
        """
        # init dp table
        t1[:, 0] = pi * b[:, 0]
        # t2[:, 0] has already initialized to 0
        # for each observation i
        for i in range(1, T):
            # for each state j
            for j in range(K):
                tmp = t1[:, i - 1] * a[:, j] * b[j, o[i]]
                t1[j, i] = np.max(tmp)
                t2[j, i] = np.argmax(tmp)
        # init z
        # x is not needed because we don't have a state space set
        z = np.zeros(T, dtype=np.int64)
        z[T - 1] = np.argmax(t1[:, T - 1])
        for i in range(T - 1, 0, -1):
            z[i - 1] = t2[z[i], i]
        return z

    def HMMBaumWelch(self, o, N, dirichlet=False, verbose=False, rand_seed=1):

        T = np.shape(o)[0]

        M = int(max(o)) + 1

        digamma = np.zeros((N, N, T))

        np.random.seed(rand_seed)

        if dirichlet:
            pi = np.ndarray.flatten(np.random.dirichlet(np.ones(N), size=1))

            a = np.random.dirichlet(np.ones(N), size=N)

            b = np.random.dirichlet(np.ones(M), size=N)
        else:

            pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N), size=1)) / 100
            pi = 1.0 / N * np.ones(N) - pi_randomizer

            a_randomizer = np.random.dirichlet(np.ones(N), size=N) / 100
            a = 1.0 / N * np.ones([N, N]) - a_randomizer

            b_randomizer = np.random.dirichlet(np.ones(M), size=N) / 100
            b = 1.0 / M * np.ones([N, M]) - b_randomizer

        error = self.tolerance + 10
        itter = 0
        while (error > self.tolerance) and (itter < self.max_iter):

            prev_a = a.copy()
            prev_b = b.copy()

            alpha = self.HMMfwd(a, b, o, pi)
            beta = self.HMMbwd(a, b, o)

            for t in xrange(T - 1):
                for i in xrange(N):
                    for j in xrange(N):
                        digamma[i, j, t] = alpha[i, t] * a[i, j] * b[j, o[t + 1]] * beta[j, t + 1]
                digamma[:, :, t] /= np.sum(digamma[:, :, t])

            for i in xrange(N):
                for j in xrange(N):
                    digamma[i, j, T - 1] = alpha[i, T - 1] * a[i, j]
            digamma[:, :, T - 1] /= np.sum(digamma[:, :, T - 1])

            for i in xrange(N):
                pi[i] = np.sum(digamma[i, :, 0])
                for j in xrange(N):
                    a[i, j] = np.sum(digamma[i, j, :T - 1]) / np.sum(digamma[i, :, :T - 1])

                for k in xrange(M):
                    filter_vals = (o == k).nonzero()
                    b[i, k] = np.sum(digamma[i, :, filter_vals]) / np.sum(digamma[i, :, :])

            error = (np.abs(a - prev_a)).max() + (np.abs(b - prev_b)).max()
            itter += 1

            if verbose:
                print("Iteration: ", itter, " error: ", error, "P(O|lambda): ", np.sum(alpha[:, T - 1]))

        return a, b, pi, alpha


def parseStockPrices(from_date, to_date, symbol):
    hist_prices = ts.get_k_data(symbol, from_date, to_date, ktype="D", autype="none")
    hist_prices_qfq = ts.get_k_data(symbol, from_date, to_date, ktype="D", autype="qfq")
    np_hist_prices = np.empty(shape=[len(hist_prices), 7])
    np_hist_prices[:, 0] = hist_prices_qfq["close"]
    np_hist_prices[:, 1] = hist_prices["close"]
    np_hist_prices[:, 2] = [datetime.datetime.strptime(dt, "%Y-%m-%d").toordinal() for dt in hist_prices.date.values]
    np_hist_prices[:, 3] = hist_prices["high"]
    np_hist_prices[:, 4] = hist_prices["low"]
    np_hist_prices[:, 5] = hist_prices["open"]
    np_hist_prices[:, 6] = hist_prices["volume"]
    return np_hist_prices


def calculateDailyMoves(hist_prices, holding_period):
    assert holding_period > 0, "Holding something less than 0 makes no sense"
    return hist_prices[:-holding_period, 1] - hist_prices[holding_period:, 1]


if __name__ == '__main__':
    hmm = myHMM()
    hist_prices = parseStockPrices('2015-10-11', '2017-11-11', '002415')
    hist_moves = calculateDailyMoves(hist_prices, 1)
    hist_Observation = np.array(list(map(lambda x: 1 if x > 0 else (0 if x < 0 else 2), hist_moves)))
    hist_Observation = hist_Observation[::-1]
    print(hist_Observation)
    (a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist_Observation, 2, False, False)
    path = hmm.HMMViterbi(a, b, hist_Observation, pi_est)
    print(path)
