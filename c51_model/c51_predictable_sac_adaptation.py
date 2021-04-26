from stable_baselines3.sac.sac import *
from c51_model.c51_predictable_policy import CMVC51SACPolicy
from stable_baselines3.common.buffers import ReplayBuffer
import time
from datetime import timedelta


class CMVCVaRSAC(OffPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[CMVC51SACPolicy]],
        env: Union[GymEnv, str],
        min_v: float = -25,
        max_v: float = +25,
        support_dim: int = 200,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = int(5e4),
        learning_starts: int = 100,
        batch_size: int = 64,
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
        cvar_alpha=0.3,
        cmv_beta=1,
    ):
        super(CMVCVaRSAC, self).__init__(
            policy,
            env,
            CMVC51SACPolicy,
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
        self._total_timesteps = None
        self.cvar_alpha = cvar_alpha
        self.cmv_beta = cmv_beta
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
        self.cmv_net = self.policy.cmv_net
        self.beta_critic = self.policy.beta_critic
        self.beta_critic_target = self.policy.beta_critic_target

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
        actor_losses, critic_losses, reward_losses, feature_pred_losses = [], [], [], []
        cvars = []
        qs = []
        critic_beta_losses = []

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

            # Compute
            # For the CMV Learning
            with th.no_grad():
                next_z = self.critic_target.features_extractor(replay_data.next_observations)
            # predicted reward, and predicted next observation
            r_pred, z_pred = self.cmv_net(replay_data.observations, replay_data.actions)
            mse_r_pred = th.mean(th.square(r_pred - replay_data.rewards))
            mse_z_pred = th.mean(th.square(z_pred - next_z))
            loss_cmv = mse_r_pred + mse_z_pred

            reward_losses.append(mse_r_pred.item())
            feature_pred_losses.append(mse_z_pred.item())

            # Optimize the CMV Nets
            self.cmv_net.optimizer.zero_grad()
            loss_cmv.backward()
            self.cmv_net.optimizer.step()

            with th.no_grad():
                # Select action according to policy
                # Compute the next Q values: min over all critics targets
                next_q_beta_values = th.cat(self.beta_critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_beta_values, _ = th.min(next_q_beta_values, dim=1, keepdim=True)
                # add entropy term
                next_q_beta_values = next_q_beta_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_beta_values = loss_cmv.detach() + (1 - replay_data.dones) * (
                            self.gamma ** 2) * next_q_beta_values

            # Get current Q-beta values estimates for each critic network
            # using action from the replay buffer
            current_q_beta_values = self.beta_critic(replay_data.observations, replay_data.actions)

            # Compute critic beta loss
            critic_beta_loss = \
                0.5 * sum(
                    [F.mse_loss(current_q_beta, target_q_beta_values) for current_q_beta in current_q_beta_values])
            critic_beta_losses.append(critic_beta_loss.item())

            # Optimize the critic beta
            self.beta_critic.optimizer.zero_grad()
            critic_beta_loss.backward()
            self.beta_critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi - qf1_beta_pi)
            # Mean over all critic networks
            z_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)

            z_cdf = th.cumsum(z_pi, dim=-1)
            adjust_pdf = th.where(
                th.le(z_cdf, self.cvar_alpha),
                z_pi,
                th.zeros_like(z_pi)
            )
            adjust_pdf = th.div(adjust_pdf, th.sum(adjust_pdf, dim=-1, keepdim=True))
            q_pi = adjust_pdf @ self.supports
            cvars.append(th.mean(q_pi).item())
            qs.append(th.mean(z_pi @ self.supports).item())
            q_beta_values_pi = th.cat(self.beta_critic.forward(replay_data.observations, actions_pi), dim=1)
            max_qf_beta_pi, _ = th.max(q_beta_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - q_pi + self.cmv_beta * next_q_beta_values).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.beta_critic.parameters(), self.beta_critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        remaining_steps = self._total_timesteps - self.num_timesteps

        eta = int(round(remaining_steps / fps))
        logger.record("time/eta", timedelta(seconds=eta), exclude="tensorboard")
        logger.record("train/CVaR Alpha", self.cvar_alpha)
        logger.record("train/CMV Beta", self.cmv_beta)
        logger.record("train/CVaR", np.mean(cvars))
        logger.record("train/Q-value", np.mean(qs))
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        logger.record("train/reward error", np.mean(reward_losses))
        logger.record("train/s_t+1_error", np.mean(feature_pred_losses))
        logger.record("train/beta_Q_loss", np.mean(critic_beta_losses))
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

        return super(CMVCVaRSAC, self).learn(
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
        return super(CMVCVaRSAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = ["log_ent_coef"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables


if __name__ == "__main__":
    from Navigation2d import NavigationEnvAcc
    from Navigation2d.config import obs_set, goal_set
    from stable_baselines3.common.vec_env import SubprocVecEnv

    import pickle


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

    env = NavigationEnvAcc({"OBSTACLE_POSITIONS": obs_set[1], "Goal": goal_set[-1]})

    rankings = []
    for _ in range(100):
        model = CMVCVaRSAC.load(f"ReptileModel")
        model.set_env(env)

        alpha = np.random.uniform(0.001, 1)
        beta = np.random.exponential(2)

        model.cvar_alpha = alpha
        model.cmv_beta = beta

        model.learn(30000)
        model.save(f"Adap({alpha},{beta})")
        logs = evaluate(env, model, 2, alpha)

        data = {
            "min_v": 25,
            "max_v": 25,
            "support_dim": 200,
            "alpha": alpha,
            "beta": beta,
            "data": logs
        }

        count = 0
        for log in logs:
            count += int(log[-1][4]["is_success"])

        rankings.append([alpha, beta, count])
        print([alpha, beta, count])

        with open(f"Adap({alpha},{beta})_logs", 'wb') as f:
            pickle.dump(data, f)

    print(sorted(rankings, key=lambda x: x[2]))
