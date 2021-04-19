from stable_baselines3.sac.sac import *
from c51 import C51SACPolicy
from stable_baselines3.common.buffers import ReplayBuffer


class CVaRSAC(OffPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[C51SACPolicy]],
        env: Union[GymEnv, str],
        min_v: float = -25,
        max_v: float = +25,
        support_dim: int = 200,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(CVaRSAC, self).__init__(
            policy,
            env,
            C51SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        self.min_v = min_v
        self.max_v = max_v
        self.support_dim = support_dim
        self.interval = (1 / (support_dim - 1)) * (max_v - min_v)
        self.supports = th.from_numpy(
            np.array([min_v + i * self.interval for i in range(support_dim)], dtype=np.float32)
        ).to(self.device)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.support_dim,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def projection(self, support_rows, target_z):
        projected_target_z = th.zeros_like(target_z)
        support_rows = support_rows.clamp(self.min_v, self.max_v - 1e-3)
        p = ((support_rows - self.min_v) % self.interval) / self.interval
        idx = ((support_rows - self.min_v) // self.interval).long()
        projected_target_z = projected_target_z.scatter_add(1, idx, target_z * p)
        projected_target_z = projected_target_z.scatter_add(1, idx + 1, target_z * (1 - p))

        return projected_target_z

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        cvars = []
        qs = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                target_zs = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                # add entropy term
                target_supports = self.supports.clone().detach()
                target_supports = target_supports - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_supports = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_supports

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_zs = th.cat(self.critic(replay_data.observations, replay_data.actions), dim=1)
            target_zs = self.projection(target_supports, target_zs)
            # Compute critic loss

            critic_loss = -th.mean(th.log(current_zs + 1e-12) * target_zs)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            z_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)

            z_cdf = th.cumsum(z_pi, dim=-1)
            adjust_pdf = th.where(
                th.le(z_cdf, 0.3),
                z_pi,
                th.zeros_like(z_pi)
            )
            adjust_pdf = th.div(adjust_pdf, th.sum(adjust_pdf, dim=-1, keepdim=True))
            q_pi = adjust_pdf @ self.supports
            cvars.append(th.mean(q_pi).item())
            qs.append(th.mean(z_pi @ self.supports).item())
            actor_loss = (ent_coef * log_prob - q_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/CVaR", np.mean(cvars))
        logger.record("train/Q", np.mean(qs))
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "C51SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(CVaRSAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(CVaRSAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = ["log_ent_coef"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables


if __name__ == "__main__":
    from stable_baselines3 import SAC
    from Navigation2d import NavigationEnvAcc, DeployEnv
    from Navigation2d.config import obs_set, goal_set
    env = NavigationEnvAcc({"OBSTACLE_POSITIONS": obs_set[1], "Goal": goal_set[-1]})

    # model = SAC(SACPolicy, env, verbose=1)
    # model = CVaRSAC(C51SACPolicy, env, min_v=-25, max_v=25, support_dim=200, verbose=1)
    # model.learn(300000)
    # model.save("CVaR-0.3")


    def evaluate(env, model, ep_n, alpha):
        logs = []
        for _ in range(ep_n):
            obs = env.reset()
            ep_reward = 0
            ep_log = []
            while True:
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, done, info = env.step(action)
                z_pi = th.cat(model.critic.forward(th.from_numpy(np.array([obs])).cuda(),
                                                   th.from_numpy(np.array([action])).cuda()), dim=1)
                z_cdf = th.cumsum(z_pi, dim=-1)
                adjust_pdf = th.where(
                    th.le(z_cdf, alpha),
                    z_pi,
                    th.zeros_like(z_pi)
                )
                adjust_pdf = th.div(adjust_pdf, th.sum(adjust_pdf, dim=-1, keepdim=True))
                cvar = adjust_pdf @ model.supports
                log = [obs, next_obs, reward, done, info,
                       z_pi.cpu().detach().numpy(), cvar.cpu().detach().numpy()]
                ep_log.append(log)
                obs = next_obs
                ep_reward += reward
                if done:
                    logs.append(ep_log)
                    break
        return logs


    import pickle
    for alpha in [0.2, 0.3, 0.4, 1.0]:
        model = CVaRSAC.load(f"CVaR-{alpha}", min_v=-25, max_v=25, support_dim=200)
        log = evaluate(env, model, 1000, alpha)
        data = {
            "min_v": 25,
            "max_v": 25,
            "support_dim": 200,
            "alpha": alpha,
            "data": log
        }
        with open(f'alpha{alpha}_ep_log.txt', 'wb') as f:
            pickle.dump(data, f)

