from stable_baselines3.common.policies import *
from stable_baselines3.sac.policies import *
from stable_baselines3.common.type_aliases import Schedule
from torch.nn import LayerNorm, ReLU, Sequential
import numpy as np


class ContinuousDistributionCritic(BaseModel):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            support_dim: int,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 1,
            share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, support_dim, net_arch, activation_fn)
            q_net.append(nn.Softmax(dim=1))
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class CMVPredictor(BaseModel):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        action_dim = get_action_dim(self.action_space)
        self.common_feature_net = nn.Sequential(*create_mlp(features_dim + action_dim, net_arch[-1],
                                                            net_arch[:-1], activation_fn))
        self.reward_net = nn.Sequential(* [nn.Linear(net_arch[-1], 2048),  nn.ReLU(),
                                           nn.Linear(2048, 1)])
        self.next_feature_pred_net = nn.Linear(net_arch[-1], features_dim,)

        self.share_features_extractor = share_features_extractor

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        common_feature_input = th.cat([features, actions], dim=1)
        feature = self.common_feature_net(common_feature_input)
        r_pred = self.reward_net(feature)
        z_pred = self.next_feature_pred_net(feature)

        return r_pred, z_pred


class LnMlpExtractor(FlattenExtractor):
    def __init__(self, observation_space: gym.Space, net_arch=None, activation_fn=ReLU):
        super(LnMlpExtractor, self).__init__(observation_space, )

        flatten_dim = int(np.prod(observation_space.shape))
        if net_arch is None:
            net_arch = [2048, 2048]

        if activation_fn == ReLU:
            activation_fn = partial(ReLU, inplace=True)

        LnMlp = create_mlp(flatten_dim, net_arch[-1], net_arch[:-1], activation_fn)
        LnMlp.append(LayerNorm(net_arch[-1]))
        LnMlp.append(activation_fn())
        LnMlp.append(nn.Linear(net_arch[-1], flatten_dim))
        self.LnMlp = Sequential(*LnMlp)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        flatten = self.flatten(observations)
        return self.LnMlp(flatten)


class CMVC51SACPolicy(BasePolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            support_dim: int,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 1,
            share_features_extractor: bool = True,
    ):
        super(CMVC51SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [256, 256]
            elif features_extractor_class == LnMlpExtractor:
                net_arch = [256, 256]
            else:
                net_arch = []

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "sde_net_arch": sde_net_arch,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
                "support_dim": support_dim,
            }
        )

        self.cmv_kwargs = self.net_args.copy()

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.beta_critic, self.beta_critic_target = None, None
        self.cmv_net = None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        # CMV net must share the feature extractor with actors.
        self.cmv_net = self.make_cmvnets(features_extractor=self.actor.features_extractor)
        cmv_parameters = [param for name, param in self.cmv_net.named_parameters() if
                          "features_extractor" not in name]
        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            self.beta_critic = self.make_beta_critic(features_extractor=self.actor.features_extractor)

            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if
                                 "features_extractor" not in name]
            beta_critic_parameters = [param for name, param in self.beta_critic.named_parameters() if
                                      "features_extractor" not in name]

        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            self.beta_critic = self.make_beta_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()
            beta_critic_parameters = self.beta_critic.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.beta_critic_target = self.make_beta_critic(features_extractor=None)
        self.beta_critic_target.load_state_dict(self.beta_critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
        self.cmv_net.optimizer = self.optimizer_class(cmv_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
        self.beta_critic.optimizer = self.optimizer_class(beta_critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                sde_net_arch=self.actor_kwargs["sde_net_arch"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousDistributionCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousDistributionCritic(**critic_kwargs).to(self.device)

    def make_beta_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self.critic_kwargs.copy()
        del critic_kwargs["support_dim"]
        critic_kwargs = self._update_features_extractor(critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def make_cmvnets(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CMVPredictor:
        cmvnet_kwargs = self._update_features_extractor(self.cmv_kwargs, features_extractor)
        return CMVPredictor(**cmvnet_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)
