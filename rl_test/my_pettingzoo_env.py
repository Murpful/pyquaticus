# my_pettingzoo_env.py
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pyquaticus.config import config_dict_std

class MyPettingZooEnv(ParallelPettingZooEnv):
    """
    This wrapper wraps your underlying PyQuaticus environment to ensure that all the
    attributes needed by your policies (for example, by BaseDefender, BaseAttacker, etc.)
    are available. For any attribute missing from the original environment, we provide
    a default value (often taken from the standard configuration).
    """
    def __init__(self, env):
        # Initialize the parent wrapper with the original env.
        super().__init__(env)
        self._orig_env = env

        # Instead of setting attributes directly (which might not be allowed),
        # we store them as properties. In many cases, your underlying env may have these,
        # but if not, we fall back to config defaults.
        # (Do not attempt to set self.agent_obs_normalizer directly if the underlying env
        #  already defines it as a read-only property.)
        
    @property
    def agent_obs_normalizer(self):
        return getattr(self._orig_env, "agent_obs_normalizer", None)

    @property
    def global_state_normalizer(self):
        return getattr(self._orig_env, "global_state_normalizer", None)
    
    @property
    def _walls(self):
        return getattr(self._orig_env, "_walls", None)
    
    @property
    def flag_keepout_radius(self):
        return getattr(self._orig_env, "flag_keepout_radius", config_dict_std.get("flag_keepout", 3.0))
    
    @property
    def catch_radius(self):
        return getattr(self._orig_env, "catch_radius", config_dict_std.get("catch_radius", 10.0))
    
    @property
    def aquaticus_field_points(self):
        return getattr(self._orig_env, "aquaticus_field_points", {})
    
    @property
    def team_size(self):
        return getattr(self._orig_env, "team_size", 1)
    
    @property
    def agents(self):
        return getattr(self._orig_env, "agents", [])
    
    @property
    def agent_ids_of_team(self):
        return getattr(self._orig_env, "agent_ids_of_team", {})
