import numpy as np

from mva_independent_component_analysis.utils.preprocessing import centering_and_whitening


def normalise_sources(sources):
    m=max([max(source) for source in sources])
    sources_=[source*(m/max(source)) for source in sources]
    return sources_


def mix_sources(sources, normalise=True):
    """
        sources : list of arrays of shape (1,length_i)
        returns : "normalized sources", mixed signals, whitened mixed signals
    """
    if normalise:
        sources=normalise_sources(sources)
    l = min([source.shape[0] for source  in sources])
    sources = np.array([source[:l] for source in sources])
    n_components, _=sources.shape
    
    while True:
        A = np.random.randint(10,100,(n_components,n_components))/10
        X = A @ sources
        # Center & whiten signals
        Xw, meanX, whiteM = centering_and_whitening(X)
        if not np.isnan(Xw).any():
            return sources, X, Xw     