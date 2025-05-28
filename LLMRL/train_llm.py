import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
import virtual_home

import json
import ast

from PPO_llm import PPO
from llmagent import LLMAgent

################################### Training ###################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(env_name, random_seed, algo, poloss_w):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 40                   # max timesteps in one episode
    max_training_timesteps = int(5e4)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e4)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 5      # update policy every n timesteps
    K_epochs = 20               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    #####################################################

    print("training environment name : " + env_name)

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]
    print("--------------------------------------------------------------------------------------------")
    print("dimension of state space : ", state_dim)

    print("Action space: ", env.action_space)

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    print("dimension of action space : ", action_dim)
    print("--------------------------------------------------------------------------------------------")

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/' + f'{algo}-{random_seed}' + "/"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = random_seed
    # current_num_files = next(os.walk(log_dir))[2]
    # run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + "/" + "result.csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################

    directory = "models" + '/' + env_name + '/' + algo + "/"
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "seed_{}.pth".format(run_num)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    print("--------------------------------------------------------------------------------------------")
    print("setting LLM policy loss weight to ", poloss_w)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, poloss_w, action_std)
    llm_agent = LLMAgent(temp=0, call_times=1, task_env=env_name)
    subgoals_str = llm_agent.subg_gen("gpt-4")
    print("Subgoals of the task: ", subgoals_str)

    subgoals_ls = json.loads(subgoals_str)
    subg_len = len(subgoals_ls)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward,episode_length\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    print_running_steps = 0

    log_running_reward = 0
    log_running_episodes = 0
    log_running_step = 0

    time_step = 0
    i_episode = 0

    tras_dic = {}
    # load transition dictionary from file
    if os.path.exists(f"tras_dic_{env_name}_test.json"):
        with open(f"tras_dic_{env_name}_test.json", "r") as f:
            tras_dic = json.load(f)
    tras_dic = {ast.literal_eval(k): v for k, v in tras_dic.items()}

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        start_obs_txt = llm_agent.task_obs2text(state)
        start_caption = start_obs_txt["prompt"]
        # print(start_obs_txt)
        start_idx = 0
        for subg in subgoals_ls:
            goal_reached, score = llm_agent.goal_reached(start_caption, subg)
            if not goal_reached:
                start_idx = subgoals_ls.index(subg)
                break
        rema_subgoals = subgoals_ls[start_idx:]
        subgoal_cur = rema_subgoals[0]

        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            subg_reward = 0

            ## generate action from LLM policy
            next_obs_txt = llm_agent.task_obs2text(state)
            act_ls = next_obs_txt["action"]
            
            caption = next_obs_txt["prompt"]

            goal_reached, score = llm_agent.goal_reached(caption, subgoal_cur)
            if goal_reached:
                rema_subgoals.pop(0)
                if len(rema_subgoals) == 0:
                    subgoal_cur = llm_agent.task_goal
                else:
                    subgoal_cur = rema_subgoals[0]
                # goal_reached, score = llm_agent.goal_reached(caption, subgoal_cur)
            if (caption, subgoal_cur) not in tras_dic:
                print("New state encountered, generating LLM policy ...")
                print("For state caption: ", caption)
                print("Subgoal: ", subgoal_cur)
                act_count_ls = [0*i for i in range(action_dim)]
                for i in range(llm_agent.call_times):
                    error = True
                    while error:
                        try:
                            llm_policy = llm_agent.policy_gen("gpt-4-turbo", act_ls, caption, subgoal_cur)
                            llm_policy = ast.literal_eval(llm_policy)
                            # choose the first action from the policy
                            llm_act_txt = llm_policy[0]
                            llm_act_id = llm_agent.action_template.index(llm_act_txt)
                            error = False
                        except Exception as e:
                            print(e)
                            print("Error, retrying ...")
                    llm_act = torch.tensor(llm_act_id).to(device)
                    act_count_ls[llm_act] += 1
                llm_act_dist = [i/llm_agent.call_times for i in act_count_ls]
                llm_act = llm_act_dist.index(max(llm_act_dist))
                llm_act = torch.tensor(llm_act).to(device)
                llm_act_txt = llm_agent.action_template[llm_act]
                print("Action from LLM: ", llm_act_txt)
                print("---------------------------------------")
                tras_dic[(caption, subgoal_cur)] = llm_act_dist
            else:
                llm_act_dist = tras_dic[(caption, subgoal_cur)]
                llm_act = llm_act_dist.index(max(llm_act_dist))
                llm_act = torch.tensor(llm_act).to(device)
                llm_act_txt = llm_agent.action_template[llm_act]

            if subg_reward > 0:
                print("The next action from LLM: ", llm_act_txt)

            if has_continuous_action_space:
                with torch.no_grad():
                    state = torch.FloatTensor(state).to(device)
                    action, action_logprob, ppo_act_dist, state_val = ppo_agent.policy_old.act(state)
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(state).to(device)
                    action, action_logprob, ppo_act_dist, state_val = ppo_agent.policy_old.act(state)

            # action = llm_act

            ppo_agent.buffer.states.append(state)
            ppo_agent.buffer.actions.append(action)
            ppo_agent.buffer.logprobs.append(action_logprob)
            ppo_agent.buffer.state_values.append(state_val)

            # save ppo and llm action distributions
            ppo_agent.buffer.llm_act_dist.append(torch.tensor(llm_act_dist).to(device))
            # save policy loss weight at this time step

            state, reward, done, _ = env.step(action)

            # if done:
            #     if reward == 1:
            #         print("Task completed!")
            #         # print("The last action: ", llm_agent.action_template[action])
            #         # with open(f"tras_dic_{env_name}_test.json", "w") as f:
            #         #     converted_dict = {str(key): value for key, value in tras_dic.items()}
            #         #     json.dump(converted_dict, f)
            #         # print("Subgoal dictionary saved in tras_dic.json")

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward+subg_reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_step = log_running_step / log_running_episodes
                log_avg_step = round(log_avg_step, 4)
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{},{}\n'.format(i_episode, time_step, log_avg_reward, log_avg_step))
                log_f.flush()

                log_running_step = 0
                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_step = print_running_steps / print_running_episodes
                print_avg_step = round(print_avg_step, 2)
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Average episode length : {}".format(i_episode, time_step, print_avg_reward, print_avg_step))

                print_running_steps = 0
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                # reset the subgoals list and current goal

                break
            
        print_running_steps += t
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_step += t
        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # save transition dictionary in a file
    with open(f"/LLMRL/tras_dic_{env_name}.json", "w") as f:
        converted_dict = {str(key): value for key, value in tras_dic.items()}
        json.dump(converted_dict, f)
    print("Subgoal dictionary saved in tras_dic.json")

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    weight_list = [1]
    seed_list = [1]

    for weight in weight_list:
        for seed in seed_list:
            train("VirtualHome-v1", seed, "LLMPPO", weight)
        
