from c51_predictable_policy import CMVC51SACPolicy
from c51_predictable_sac import CMVCVaRSAC
from sac_cvar import CVaRSAC
from c51 import C51SACPolicy
from stable_baselines3 import SAC
from stable_baselines3.sac.sac import SACPolicy
import torch as th
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
from Navigation2d import NavigationEnvAcc
from Navigation2d.config import obs_set, goal_set
import json
import pickle



def evaluate(env, model, ep_n=1000):
    logs = []
    for _ in tqdm(range(ep_n)):
        obs = env.reset()
        ep_reward = 0
        ep_log = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)

            z_pi = th.cat(model.critic.forward(th.from_numpy(obs).to(model.device),
                                               th.from_numpy(action).to(model.device)), dim=1)
            z_cdf = th.cumsum(z_pi, dim=-1)
            adjust_pdf = th.where(
                th.le(z_cdf, model.cvar_alpha),
                z_pi,
                th.zeros_like(z_pi)
            )
            adjust_pdf = th.div(adjust_pdf, th.sum(adjust_pdf, dim=-1, keepdim=True))
            cvar = adjust_pdf @ model.supports.to(model.device)
            log = [obs, next_obs, reward, done, info,
                   z_pi.cpu().detach().numpy(), cvar.cpu().detach().numpy()]
            ep_log.append(log)
            obs = next_obs
            ep_reward += reward
            if done:
                logs.append(ep_log)
                break

    return logs






if __name__ == "__main__":
    if __name__ == "__main__":

        env = SubprocVecEnv(
            [lambda: NavigationEnvAcc({"OBSTACLE_POSITIONS": obs_set[1], "Goal": goal_set[-1]}) for _ in range(1)])
        # model = SAC(SACPolicy, env, verbose=1)
        hyperparameters = dict(
            min_v=-25,
            max_v=25,
            support_dim=200,
            verbose=1,
            batch_size=256,
            buffer_size=int(2e5),
        )

        save_prefix = "/workspace/files/models/"
        log_prefix = "/workspace/files/logs/"
        learn_steps = int(1e+6)
        with open("hyperparams.json", "w") as f:
            json.dump(hyperparameters, f)


        def learn_and_save(model_class, policy_class, env, name, cvar_param, model_kwargs=None):
            if model_kwargs is None:
                model = model_class(policy_class, env, cvar_alpha=alpha, **hyperparameters)
            else:
                import copy
                hparam = copy.deepcopy(hyperparameters)
                hparam.update(model_kwargs)
                model = model_class(policy_class, env, cvar_alpha=alpha, **hparam)

            model.learn(learn_steps)
            model.save(save_prefix + f"/{name}_{cvar_param}")
            log = evaluate(env, model)
            with open( log_prefix + f"{name}_{cvar_param}.pkl", "wb") as f:
                data = {
                    "data": log,
                    "alpha":alpha
                }
                pickle.dump(data, f)


        for alpha in [0.2, 0.3, 0.4, 1.0]:

            learn_and_save(CMVCVaRSAC, CMVC51SACPolicy, env, cvar_param=alpha, name="pred_all")
            learn_and_save(CMVCVaRSAC, CMVC51SACPolicy, env, cvar_param=alpha, name="state_only", model_kwargs=dict(minimize_cmv_reward=False,
        minimize_cmv_state=True))
            learn_and_save(CMVCVaRSAC, CMVC51SACPolicy, env, cvar_param=alpha, name="reward_only", model_kwargs=
            dict(minimize_cmv_reward=True,
                 minimize_cmv_state=False)
                           )
            model = CVaRSAC(C51SACPolicy, env, cvar_alpha=alpha, **hyperparameters)
            model.learn(learn_steps)
            model.save(save_prefix + f"/raw_cvar_{alpha}")
            log = evaluate(env, model)
            with open( log_prefix + f"raw_cvar_{alpha}.pkl", "wb") as f:
                data = {
                    "data": log,
                    "alpha":alpha
                }
                pickle.dump(data, f)
            del model
            del log



