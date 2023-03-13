from risk_measures.measures import *

risk_map = {"neutral": neutral,
            "averse_cvar": averse_cvar,
            "seeking_cvar": seeking_cvar,
            "power": power,
            "wang": wang,
            "cpw": cpw}


# just utility for the one shot experiment
class EnumerateRisk(object):
    def __init__(self):
        self.risk_keys = list(risk_map.keys())
        self.risk_keys.sort()
        self.risk_args = {"averse_cvar": [0.5, 0.75],
                          "seeking_cvar": [0.1],
                          "power": [-0.5, 0.5],
                          "wang": [-0.5, 0.5],
                          "cpw": [0.7, -0.7],
                          "neutral": [()]
                          }

    def __iter__(self):
        for k in self.risk_keys:
            risk_args = self.risk_args[k]
            for a in risk_args:
                yield {"name": k, "function": risk_map[k], "arg": a}

