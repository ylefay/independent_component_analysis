import numpy as np
from mva_independent_component_analysis.utils.metrics import fast_corr_coef


def fast_ica(signals, alpha=1, thresh=1e-8, iterations=5000, true_sources=None):
    m, n = signals.shape  # (m,n)

    # Initialize random weights
    W = np.random.rand(m, m)

    mcc = []
    W_hat = W.copy()
    for c in range(m):
        last_distance = 10000
        print(f'component {c}')
        w = W[c, :].copy()  # (1,p)
        w = w / np.sqrt((w ** 2).sum())
        i = 0
        while (i < iterations):

            # Dot product of weight and signal
            wTx = np.dot(w, signals)  # (1,p)(p,n)=(1,n)

            # Pass w*s into contrast function g
            gwTx = np.tanh(wTx * alpha).T  # (n,1)

            # Pass w*s into g prime
            g_wTx = (1 - np.square(np.tanh(wTx))) * alpha  # (n,1)

            # Update weights
            w_ = (signals * gwTx.T).mean(axis=1) - g_wTx.mean() * w.squeeze()

            # Decorrelate weights
            w_ = w_ - np.dot(np.dot(w_, W[:c].T), W[:c])

            # Normalize
            w_ = w_ / np.sqrt((w_ ** 2).sum())

            w_ /= np.linalg.norm(w_)

            distance = np.abs(np.abs(w_.T @ w) - 1)
            print(f'iteration : {i} : distance {distance}')
            # ~~~~~~inference

            if true_sources is not None:
                W_hat[c, :] = w_
                unMixed_hat = W_hat @ signals
                print(f'----- MCC = {fast_corr_coef(true_sources, unMixed_hat)}')
                mcc.append(fast_corr_coef(true_sources, unMixed_hat))

            if distance < thresh:
                break
            elif distance == last_distance:
                break

            last_distance = distance
            # Update weights
            w = w_

            # Update counter
            i += 1

        W[c, :] = w
    return W, mcc
