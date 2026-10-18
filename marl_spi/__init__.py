from .actions import Action, Command
from .observation import LocalObservation, Neighbor, Role
from importlib import import_module

__all__ = ["Action", "Command", "LocalObservation", "Neighbor", "Role", "LocalAgent", "create_actor_critic_agent", "LocalQAgent", "create_q_agent"]


def __getattr__(name):
    modules = {"LocalAgent": ".agent", "create_actor_critic_agent": ".agent", "LocalQAgent": ".q_agent", "create_q_agent": ".q_agent"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
