from stable_baselines3.common.callbacks import BaseCallback
import time
import zmq
import pickle as pkl
import torch
import numpy as np


class LowCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, oper_num, port, verbose=0):
        super(LowCallback, self).__init__(verbose)
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.connect('tcp://localhost:{}'.format(port))
        self.request_msg = {'operator_number': oper_num, 'description': 'request'}
        self.recv_sock = ctx.socket(zmq.SUB)
        self.recv_sock.connect('tcp://localhost:{}'.format(port+1))
        self.recv_sock.setsockopt_string(zmq.SUBSCRIBE, '')
        self.update = 0

    def _on_step(self):
        return True

    def _on_training_start(self) -> None:
        res = pkl.loads(self.recv_sock.recv())
        if res['description'] == 'parameters':
            model_parameter = res['parameters']
            self.model.set_parameters(model_parameter)

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.update += 1
        if self.update == 64:
            self.update = 0
            model_parameter = self.model.get_parameters()
            msg = {'operator_number': self.request_msg['operator_number'], 'description': 'parameters',
                   'parameters': model_parameter}
            self.sock.send(pkl.dumps(msg))
            _ = pkl.loads(self.sock.recv())
            res = pkl.loads(self.recv_sock.recv())
            if res['description'] == 'parameters':
                model_parameter = res['parameters']
                self.model.set_parameters(model_parameter)
                self.model.cmv_beta = np.random.exponential(2)
                self.model.cvar_alpha = np.random.uniform(0.001, 1)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        msg = {'operator_number': self.request_msg['operator_number'], 'description': 'finish'}
        self.sock.send(pkl.dumps(msg))


class Reptile(object):
    def __init__(self, num_of_operator, port, alpha, env, model):
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REP)
        self.sock.bind('tcp://*:{}'.format(port))
        self.send_sock = ctx.socket(zmq.PUB)
        self.send_sock.bind('tcp://*:{}'.format(port+1))
        self.num_of_operator = num_of_operator
        self.response_msg = {'description': 'response'}
        self.alpha = alpha
        self.model = model
        self.env = env
        self.test_model = None
        time.sleep(1.0)
        model_parameter = self.model.get_parameters()
        msg = {'description': 'parameters', 'parameters': model_parameter}
        self.send_sock.send(pkl.dumps(msg))

    def run(self):
        while True:
            num_data = 0
            data = dict()
            while num_data != self.num_of_operator:
                req = pkl.loads(self.sock.recv())
                if req['description'] == 'parameters':
                    data[str(req['operator_number'])] = req
                    num_data += 1
                if req['description'] == 'finish':
                    return
                self.sock.send(pkl.dumps(self.response_msg))
            model_parameter = self.model.get_parameters()
            parameter = model_parameter['policy']
            for layer in parameter.keys():
                layer_param = []
                for i in range(self.num_of_operator):
                    layer_param.append(data[str(i)]['parameters']['policy'][layer])
                delta = torch.mean(torch.stack(layer_param), 0)
                parameter[layer] = (1 - self.alpha) * parameter[layer] + self.alpha * delta
            self.model.set_parameters(model_parameter)
            model_parameter = self.model.get_parameters()
            msg = {'description': 'parameters', 'parameters': model_parameter}
            self.send_sock.send(pkl.dumps(msg))

    def test(self):
        sum_of_reward = 0
        parameters = self.model.get_parameters()
        self.model.learn(total_timesteps=1024)
        for i in range(10):
            done = False
            state = self.env.reset()
            while not done:
                action, _ = self.model.predict(state)
                state, reward, done, info = self.env.step(action)
                sum_of_reward += reward
        print("adaptation score:", sum_of_reward / 10)
        self.model.set_parameters(parameters)
        return

    def save(self, path):
        self.model.save(path)

    def adapt(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
