# -*- coding: UTF-8 -*-

ERROR_INFO_LIMIT = 100 # limit 100 characters

class SubmissionFolderNotExistException(Exception):
    def __str__(self):
        return "`submission` folder cannot be found in your submitted zip file."


class AgentFileNotExistException(Exception):
    def __str__(self):
        return "`agent.py` cannot be found in your submission"

class AgentClassCannotImportException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        message = "[`Agent` class in agent.py cannot be imported] {}".format(self.error_info)
        return message[:ERROR_INFO_LIMIT]

class AgentInitException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        message = "[Agent init failed] {}".format(self.error_info)
        return message[:ERROR_INFO_LIMIT]

class AgentActException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        message = "[Agent act failed] {}".format(self.error_info)
        return message[:ERROR_INFO_LIMIT]

class EnvStepException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        message = "[Env step failed] {}".format(self.error_info)
        return message[:ERROR_INFO_LIMIT]

class AgentActTimeout(Exception):
    def __str__(self):
        message = "[Agent act timeout] executing the act function of your agent is out of time limit."
        return message[:ERROR_INFO_LIMIT]

class EvaluationRunTimeout(Exception):
    def __str__(self):
        message = "[Evaluation run timeout] the evaluation of your submission is out of time limit."
        return message[:ERROR_INFO_LIMIT]


#### timeout utils
import signal

class TimeoutException(Exception):
    def __str__(self):
        return "timeout exception."

def handler(signum, frame):
    raise TimeoutException()

class TimeoutContext(object):
    def __init__(self, timeout_s):
        """ Only supported in UNIX

        Args:
            timeout_s(int): seconds of timeout limit
        """
        assert isinstance(timeout_s, int)
        signal.signal(signal.SIGALRM, handler)
        self.timeout_s = timeout_s

    def __enter__(self):
        signal.alarm(self.timeout_s)

    def __exit__(self, type, value, tb):
        # Cancel the timer if the function returned before timeout
        signal.alarm(0)
####


import os
import copy
import numpy as np


from environment.base_env import Environment

def run_one_episode(env, seed, start_timestep, episode_max_steps, agent, act_timeout):
    obs = env.reset(seed=seed, timestep=start_timestep)

    reward = 0.0
    done = False

    sum_reward = 0.0
    sum_steps = 0.0
    act_timeout_context = TimeoutContext(act_timeout)
    for step in range(episode_max_steps):
        try:
            with act_timeout_context:
                action = agent.act(obs, reward, done)
        except Exception as e:
            if isinstance(e, TimeoutException):
                raise AgentActTimeout()
            raise AgentActException(str(e))

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            if isinstance(e, TimeoutException):
                raise TimeoutException()
            raise EnvStepException(str(e))

        sum_reward += reward
        sum_steps += 1

        if done:
            break

    return sum_reward, sum_steps

def update_evaluate_settings(settings, evaluate_data_path):
    settings.load_p_filepath = os.path.join(evaluate_data_path, 'load_p.csv')
    settings.gen_p_filepath = os.path.join(evaluate_data_path, 'gen_p.csv')
    settings.load_p_forecast_filepath = os.path.join(evaluate_data_path, 'load_p_forecast.csv')
    settings.gen_p_forecast_filepath = os.path.join(evaluate_data_path, 'gen_p_forecast.csv')

    return settings

def eval():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_episodes', type=int, required=True, help='Episodes number of the evaluation.')
    parser.add_argument('--seed', type=int, required=True, help='Seed of the evaluation.')
    parser.add_argument('--evaluate_data_path', type=str, required=True, help='Path of the evaluation data.')
    parser.add_argument('--act_timeout', type=int, required=True, help='time limit (seconds) of agent act function.')
    parser.add_argument('--run_timeout', type=int, required=True, help='time limit (seconds) of the whole evaluation.')
    args = parser.parse_args()

    this_directory_path = os.path.dirname(os.path.abspath(__file__))
    this_directory_path = os.path.join(this_directory_path, 'submission') # submission directory

    from utilize.settings import settings

    if not os.path.isdir("submission"):
        raise SubmissionFolderNotExistException()
    if not os.path.isfile("submission/agent.py"):
        raise AgentFileNotExistException()

    try:
        from submission.agent import Agent
    except Exception as e:
        raise AgentClassCannotImportException(str(e))
    
    run_timeout_context = TimeoutContext(args.run_timeout)
    
    try:
        with run_timeout_context:
            try:
                agent = Agent(copy.deepcopy(settings), this_directory_path)
            except Exception as e:
                if isinstance(e, TimeoutException):
                    raise TimeoutException()
                raise AgentInitException(str(e))

            # NOTE: update evaluation settings after passing settings to the user agent.
            settings = update_evaluate_settings(settings, args.evaluate_data_path)

            env = Environment(settings)

            start_timestep = 1
            episode_max_steps = 288
            scores = []
            for i in range(args.num_episodes):
                score = run_one_episode(env, args.seed + i, start_timestep, episode_max_steps, agent, args.act_timeout)
                scores.append(score)

                start_timestep += episode_max_steps # next episode (E.g. next day)
    
    except Exception as e:
        if isinstance(e, TimeoutException):
            raise EvaluationRunTimeout()
        else:
            raise e

    mean_score = np.mean(scores)

    return {'score': mean_score}

if __name__ == "__main__":
    try:
        score = eval()['score']
        print('[Succ]')
        print('Score = %.4f' % score)
    except Exception as e:
        print('[Fail]')
        print(type(e))
        print(e)
        print('-'*50 + '\n')
