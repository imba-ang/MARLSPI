from .hybrid import CountPriorBuilder, FrozenPrior


def build_empirical_prior(samples) -> FrozenPrior:
    builder = CountPriorBuilder()
    builder.extend(samples)
    return builder.freeze()
