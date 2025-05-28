from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import numpy as np
import nltk
import re

# 下载punkt标记器
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


class LLMAgent():
    def __init__(self, temp, call_times, task_env):
        self.temp = temp
        self.call_times = call_times
        self.client = OpenAI()
        self.task_env = task_env
        self.sub_sys_promp = "You are helper to define subgoals for an Agent in order to complete long horizon tasks. Based on the given task description and environmental information, you are required to output a sequence of goals that the Agent needs to achieve in order.\nRemember:\n1. The subgoals should depict the states of objects in the environment instead of actions to do.\n2. Output the subgoals in a list, e.g. [subgoal1, subgoal2, ...].\n3. Try to generate as less subgoals as you can and you do not need to make them detailed steps.\n4. You should output the subgoal as simple as possible without any explanation."
        self.poli_sys_promp = "You are helper to generate policies for an agent to reach subgoals in completing long horizon tasks. Based on the environment information, given current state of the environment and the subgoal for the moment, you are required to output a sequence of actions to reach this goal.\nRemember: You have to store the action sequence in a [](list) type."
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        if self.task_env == "VirtualHome-v1":
            self.action_template = [
                "walk to the living room",  # 0
                "walk to the kitchen",  # 1
                "walk to the bathroom",  # 2
                "walk to the bedroom",  # 3

                "walk to the pancake",  # 4
                "walk to the microwave",  # 5

                "grab the pancake",  # 6
                "put the pancake in the microwave",  # 7
                'open the microwave',  # 8
                'close the microwave',  # 9
            ]
            self.task_goal = "In this setting, there is an Agent in a virtual home environment. The task is to prepare food, specifically, the Agent needs to get the pancake and put it in the microwave which are both located in the kitchen."
        if self.task_env == "VirtualHome-v2":
            self.action_template = [
                "walk to the living room", # 0
                "walk to the kitchen", # 1
                "walk to the bathroom", # 2
                "walk to the bedroom", # 3

                "walk to the chips", # 4
                "walk to the milk", # 5
                "walk to the coffee table", # 6
                "walk to the TV", # 7
                "walk to the sofa", # 8

                "grab the chips", # 9
                "grab the milk", # 10
                "put the chips on the coffee table", # 11
                "put the milk on the coffee table", # 12
                "turn on the TV", # 13
                "turn off the TV", # 14
                "sit on the sofa", # 15
                "stand up from the sofa" # 16
            ]
            self.task_goal = "In this setting, there is an Agent in a virtual home environment. The task is to relax in the living room. Specifically, the Agent needs to get the chips and the milk from the kitchen, then turn on the TV and sit on the sofa in the living room."

    def subg_gen(self, model):
        subg_promp =[
                {
                "role": "system",
                "content": self.sub_sys_promp
                },
                {
                "role": "user",
                "content": self.task_goal
                }
        ]
        subgs = self.client.chat.completions.create(
            model=model,
            messages=subg_promp,
            temperature=self.temp,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return subgs.choices[0].message.content

    def policy_gen(self, model, acts_list, capt, subg):
        if self.task_env == "VirtualHome-v1":
            user_req = self.task_goal + f"\n\nThe available actions are as follows:\n{str(acts_list)}\n\nThe information about the environment:\n1. There are four rooms in the home, living room, kitchen, bathroom and bedroom.\n2. The Agent initially holds nothing in hands.\n3. The pancake and the microwave are located in the kitchen.\n\nAnswer template:\nAnalysis: <brief the reason to do so>\nAction sequence: [act1, act2, ...]\n\nThe caption of the current state is:\n{capt}\n\nIn order to achieve subgoal: {subg}, what should you do next?"
            policy_promp = [
                {
                "role": "system",
                "content": self.poli_sys_promp
                },
                {
                "role": "user",
                "content": user_req
                }
            ]
        elif self.task_env == "VirtualHome-v2":
            user_req = self.task_goal + f"\n\nThe available actions are as follows:\n{str(acts_list)}\n\nThe information about the environment:\n1. There are four rooms in the home, living room, kitchen, bathroom and bedroom.\n2. The agent initially holds nothing in hands.\n3. The chips and the milk are located in the kitchen, the sofa and TV are located in the living room.\n4. Agent has two hands and can hold one object in each hand, consider if the Agent needs to free one before interaction.\n5. Agent needs to get near the object before using it.\n\nAnswer template:\nAnalysis: <brief the reason to do so>\nAction sequence: [act1, act2, ...]\n\nThe caption of the current state is:\n{capt}\n\nIn order to achieve subgoal: {subg}, what should you do next?"
            policy_promp = [
                {
                "role": "system",
                "content": self.poli_sys_promp
                },
                {
                "role": "user",
                "content": user_req
                }
            ]
        # print(user_req)
        policy = self.client.chat.completions.create(
            model=model,
            messages=policy_promp,
            temperature=self.temp,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            max_tokens=256
        )
        policy_content = policy.choices[0].message.content
        print(policy_content)
        act_match = re.search(r'Action sequence:\s*(\[[^\]]*\])', policy_content, re.DOTALL)
        act_seq = act_match.group(1)
        # print(act_seq)

        return act_seq
    
    
    def task_obs2text(self, obs):

        if self.task_env == "VirtualHome-v1":

            text = ""

            in_kitchen = obs[0]
            in_bathroom = obs[1]
            in_bedroom = obs[2]
            in_livingroom = obs[3]

            see_pancake = obs[4]
            close_to_pancake = obs[5]
            hold_pancake = obs[6]

            see_microwave = obs[7]
            close_to_microwave = obs[8]
            is_microwave_open = obs[9]

            pancake_in_microwave = obs[10]

            assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

            # template for room
            # in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {}. "
            in_room_teplate = "Agent is in the {}. "
            if in_kitchen:
                text += in_room_teplate.format("kitchen")
            elif in_bathroom:
                text += in_room_teplate.format("bathroom")
            elif in_bedroom:
                text += in_room_teplate.format("bedroom")
            elif in_livingroom:
                text += in_room_teplate.format("living room")

            object_text = ""

            if in_kitchen:

                if not see_pancake or pancake_in_microwave:
                    object_text += "Pancake is in the microwave. "
                else:
                    object_text += "Agent notices pancake and microwave. "

                # if close_to_microwave and not hold_pancake:
                #     object_text += "Pancake is in the microwave. "

                if hold_pancake:
                    object_text += "Currently, Agent has the pancake. "
                    if close_to_microwave:
                        object_text += "The microwave is nearby. "

                    else:
                        object_text += "The microwave is not nearby. "

                else:
                    if close_to_pancake and not close_to_microwave:
                        object_text += "Currently, Agent is not grabbing anything in hands. The pancake is nearby. "

                    elif close_to_microwave and not close_to_pancake:
                        object_text += "Currently, Agent is not grabbing anything in hands. The microwave is nearby. "

                    elif not close_to_pancake and not close_to_microwave:
                        object_text += "Currently, Agent is not grabbing anything in hands. The pancake and the microwave are not nearby. "
                    # else:
                    #     if is_microwave_open:
                    #         action_list = [0, 2, 3, 8, 9]
                    #     else:
                    #         action_list = [0, 2, 3, 9]

                if see_pancake and is_microwave_open:
                    object_text += "The microwave is opened. "
                elif see_pancake and not is_microwave_open:
                    object_text += "The microwave is not opened. "
                else:
                    object_text += "The microwave is closed. "

            elif in_bathroom:

                if hold_pancake:
                    object_text += "and notices nothing useful. Currently, Agent has the pancake. "
                else:
                    object_text += "and notices nothing useful. Currently, Agent is not grabbing anything in hands. "

            elif in_bedroom:

                if hold_pancake:
                    object_text += "and notices nothing useful. Currently, Agent has the pancake. "
                else:
                    object_text += "and notices nothing useful. Currently, Agent is not grabbing anything in hands. "

            elif in_livingroom:

                if hold_pancake:
                    object_text += "and notices nothing useful. Currently, Agent has the pancake. "
                else:
                    object_text += "and notices nothing useful. Currently, Agent is not grabbing anything in hands. "

            text += object_text

            self.template2action = {
                k:i for i,k in enumerate(self.action_template)
            }

            action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            actions = [self.action_template[i] for i in action_list]

        elif self.task_env == "VirtualHome-v2":

            text = ""

            in_kitchen = obs[0]
            in_bathroom = obs[1]
            in_bedroom = obs[2]
            in_livingroom = obs[3]
            
            see_chips = obs[4]
            close_to_chips = obs[5]
            hold_chips = obs[6]
            chips_on_coffeetable = obs[7]
            
            see_milk = obs[8]
            close_to_milk = obs[9]
            hold_milk = obs[10]
            milk_on_coffeetable = obs[11]

            see_tv = obs[12]
            close_to_tv = obs[13]
            is_face_tv = obs[14]
            is_tv_on = obs[15]

            see_sofa = obs[16]
            close_to_sofa = obs[17]
            is_sit_sofa = obs[18]

            see_coffeetable = obs[19]
            close_to_coffeetable = obs[20]
            assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

            # template for room
            # in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. Agent is in the {} "
            in_room_teplate = "Agent is in the {} "
            if in_kitchen:
                text += in_room_teplate.format("kitchen")
            elif in_bathroom:
                text += in_room_teplate.format("bathroom")
            elif in_bedroom:
                text += in_room_teplate.format("bedroom")
            elif in_livingroom:
                text += in_room_teplate.format("living room")

            ########################################template2####################################
            # template for kitchen
            object_text = ""

            if in_kitchen:

                if see_chips and see_milk:
                    object_text += "and notices chips and milk. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, Chips and milk are in Agent's hands. "

                    elif hold_chips and not hold_milk:
                        if close_to_milk:
                            object_text += "The milk is near Agent. But Agent has not grabbed the milk. Currently, Agent has grabbed the chips in hand. "

                        else:
                            object_text += "The milk is not near Agent. Currently, Agent has grabbed the chips in hand. "

                    elif not hold_chips and hold_milk:
                        if close_to_chips:
                            object_text += "The chips are near Agent. But Agent has not grabbed the chips. Currently, Agent has grabbed the milk in hands. "

                        else:
                            object_text += "The chips are not near Agent. Currently, Agent has grabbed the milk in hand. "

                    else:
                        if close_to_chips and close_to_milk:
                            object_text += "They are near Agent. But Agent has not grabbed the them. "

                        elif close_to_chips and not close_to_milk:
                            object_text += "The chips are near Agent. But Agent has not grabbed the chips. "

                        elif not close_to_chips and close_to_milk:
                            object_text += "The milk is near Agent. But Agent has not grabbed the milk. "

                        else:
                            object_text += "But they are not close to Agent. "

                        object_text += "Currently, Agent is not grabbing anything in hands. "

                elif see_chips and not see_milk:
                    object_text += "and only notices chips. "

                    if hold_chips:
                        object_text += "Currently, Agent has grabbed the chips in hand. "

                    else:
                        if close_to_chips:
                            object_text += "The chips are near Agent. But Agent has not grabbed the chips. "
                            
                        else:
                            object_text += "The chips are not near Agent. "

                elif not see_chips and see_milk:
                    object_text += "and notices milk. "

                    if hold_milk:
                        object_text += "Currently, Agent has grabbed the milk in hand. "

                    else:
                        if close_to_milk:
                            object_text += "The milk is near Agent. But Agent has not grabbed the milk. "

                        else:
                            object_text += "The milk is not near Agent. "

                else:
                    object_text += "and notices nothing. "

            elif in_livingroom:

                object_text += "and Agent notices a coffee table, a TV and a sofa. "

                assert close_to_coffeetable + close_to_tv + close_to_sofa <= 1, "Agent is next to more than one object from coffee table, TV and sofa."
                assert see_coffeetable + see_tv + see_sofa >= 3, "Agent does not see coffee table, TV and sofa."

                if not close_to_coffeetable and not close_to_tv and not close_to_sofa:
                    object_text += "They are not near Agent. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, Chips and milk are in Agent's hands. "
                    elif not hold_chips and hold_milk:
                        object_text += "Currently, Agent has grabbed the milk in hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, Agent has grabbed the chips in hand. "
                    else:
                        object_text += "Currently, Agent is not grabbing anything in hands. "

                if close_to_coffeetable:

                    if (chips_on_coffeetable and hold_milk) or (milk_on_coffeetable and hold_chips):
                        object_text += "The TV is not near Agent. "
                    else:
                        object_text += "The coffee table is near Agent. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, Chips and milk are in Agent's hands. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, Agent has grabbed the milk in hand. "

                        else:
                            object_text += "Currently, chips on the coffee table and the milk in the hand. "

                    elif hold_chips and not hold_milk:
                        object_text += "Currently, Agent has grabbed the chips in hand. "

                        if not milk_on_coffeetable:
                            object_text += "Currently, Agent has grabbed the chips in hand. "

                        else:
                            object_text += "Currently, milk on the coffee table and the chips in the hand. "

                    else:
                        object_text += "Currently, Agent is not grabbing anything in hands. "

                if close_to_tv:
                    if is_tv_on:
                        object_text += "The sofa is not near Agent. "

                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on. Chips and milk are in Agent's hands. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on. Agent has grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on. Chips on the coffee table and the milk in the hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on. Agent has grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on. Agent has grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on. Milk on the coffee table and the chips in the hand. "
                        else:
                            object_text += "Currently, the TV is turned on. Agent is not grabbing anything in hands with chips and milk on the coffee table. "

                    else:
                        object_text += "The TV is near Agent. "

                        if hold_chips and hold_milk:
                            object_text += "Currently, Chips and milk are in Agent's hands. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, Agent has grabbed the milk in hand. "
                            else:
                                object_text += "Currently, chips on the coffee table and the milk in the hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, Agent has grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, Agent has grabbed the chips in hand. "
                            else:
                                object_text += "Currently, milk on the coffee table and the chips in the hand. "

                if close_to_sofa:

                    if not is_sit_sofa:
                        object_text += "The sofa is near Agent. "

                        if is_tv_on:
                            if hold_chips and hold_milk:
                                object_text += "Currently, the TV is turned on. Chips and milk are in Agent's hands. "

                            elif not hold_chips and hold_milk:
                                if not chips_on_coffeetable:
                                    object_text += "Currently, the TV is turned on. Agent has grabbed the milk in hand. "
                                else:
                                    object_text += "Currently, the TV is turned on. Chips on the coffee table and the milk in the hand. "
                            elif hold_chips and not hold_milk:
                                object_text += "Currently, the TV is turned on. Agent has grabbed the chips in hand. "
                                if not milk_on_coffeetable:
                                    object_text += "Currently, the TV is turned on. Agent has grabbed the chips in hand. "
                                else:
                                    object_text += "Currently, the TV is turned on. Milk on the coffee table and the chips in the hand. "
                            else:
                                object_text += "Currently, the TV is turned on. Agent is not grabbing anything in hands with chips and milk on the coffee table. "

                        else:
                            if hold_chips and hold_milk:
                                object_text += "Currently, Chips and milk are in Agent's hands. "

                            elif not hold_chips and hold_milk:
                                if not chips_on_coffeetable:
                                    object_text += "Currently, Agent has grabbed the milk in hand. "
                                else:
                                    object_text += "Currently, chips on the coffee table and the milk in the hand. "
                            elif hold_chips and not hold_milk:
                                object_text += "Currently, Agent has grabbed the chips in hand. "
                                if not milk_on_coffeetable:
                                    object_text += "Currently, Agent has grabbed the chips in hand. "
                                else:
                                    object_text += "Currently, milk on the coffee table and the chips in the hand. "

                    else:
                        object_text += "Agent is sitting on the sofa. "

                        if is_tv_on:
                            if hold_chips and hold_milk:
                                object_text += "Currently, the TV is turned on. Chips and milk are in Agent's hands. "

                            elif not hold_chips and hold_milk:
                                if not chips_on_coffeetable:
                                    object_text += "Currently, the TV is turned on. Agent has grabbed the milk in hand. "
                                else:
                                    object_text += "Currently, the TV is turned on. Chips on the coffee table and the milk in the hand. "
                            elif hold_chips and not hold_milk:
                                object_text += "Currently, the TV is turned on. Agent has grabbed the chips in hand. "
                                if not milk_on_coffeetable:
                                    object_text += "Currently, the TV is turned on. Agent has grabbed the chips in hand. "
                                else:
                                    object_text += "Currently, the TV is turned on. Milk on the coffee table and the chips in the hand. "

                        else:
                            if hold_chips and hold_milk:
                                object_text += "Currently, Chips and milk are in Agent's hands. "

                            elif not hold_chips and hold_milk:
                                if not chips_on_coffeetable:
                                    object_text += "Currently, Agent has grabbed the milk in hand. "
                                else:
                                    object_text += "Currently, chips on the coffee table and the milk in the hand. "
                            elif hold_chips and not hold_milk:
                                object_text += "Currently, Agent has grabbed the chips in hand. "
                                if not milk_on_coffeetable:
                                    object_text += "Currently, Agent has grabbed the chips in hand. "
                                else:
                                    object_text += "Currently, milk on the coffee table and the chips in the hand. "

            elif in_bedroom:

                if hold_chips and hold_milk:
                    object_text += "and notices nothing. Currently, Chips and milk are in Agent's hands. "
                elif hold_chips and not hold_milk:
                    object_text += "and notices nothing. Currently, Agent has grabbed the chips in hand. "
                elif not hold_chips and hold_milk:
                    object_text += "and notices nothing. Currently, Agent has grabbed the milk in hand. "
                else:
                    object_text += "and notices nothing. Currently, Agent is not grabbing anything in hands. "

            elif in_bathroom:

                if hold_chips and hold_milk:
                    object_text += "and notices nothing. Currently, Chips and milk are in Agent's hands. "
                elif hold_chips and not hold_milk:
                    object_text += "and notices nothing. Currently, Agent has grabbed the chips in hand. "
                elif not hold_chips and hold_milk:
                    object_text += "and notices nothing. Currently, Agent has grabbed the milk in hand. "
                else:
                    object_text += "and notices nothing. Currently, Agent is not grabbing anything in hands. "

            text += object_text

            self.template2action = {
                k:i for i,k in enumerate(self.action_template)
            }

            action_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            actions = [self.action_template[i] for i in action_list]

        else:
            raise ValueError("Invalid task environment")
        
        return {"prompt": text, "action": actions}
        
        
    def goal_reached(self, caption, subgoal):
        state_sentences = sent_tokenize(caption)

        subgoal_embedding = self.model.encode(subgoal, convert_to_tensor=True)

        similarity_threshold = 0.9
        subgoal_achieved = False

        for sentence in state_sentences:
            sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
            cosine_similarity = util.pytorch_cos_sim(sentence_embedding, subgoal_embedding)

            if cosine_similarity >= similarity_threshold:
                # print("Similarity:", cosine_similarity.item())
                subgoal_achieved = True
                break

        return subgoal_achieved, cosine_similarity
            
        
    
# if __name__ == '__main__':
#     Agent = LLMAgent(temp=0, call_times=5, task_env="VirtualHome-v2")
#     # subgs = Agent.subg_gen("gpt-4")
#     # print("Subgoals for the task:\n", subgs)
#     act = [
#                 "walk to the living room", # 0
#                 "walk to the kitchen", # 1
#                 "walk to the bathroom", # 2
#                 "walk to the bedroom", # 3

#                 "walk to the chips", # 4
#                 "walk to the milk", # 5
#                 'walk to the coffee table', # 6
#                 'walk to the TV', # 7
#                 'walk to the sofa', # 8

#                 "grab the chips", # 9
#                 "grab the milk", # 10
#                 'put the chips on the coffee table', # 11
#                 'put the milk on the coffee table', # 12
#                 "turn on the TV", # 13
#                 "turn off the TV", # 14
#                 "sit on the sofa", # 15
#                 "stand up from the sofa" # 16
#             ]
#     capt = "Agent is in the kitchen and notices chips and milk. The milk is near Agent. But Agent has not grabbed the milk. Currently, Agent has grabbed the chips in hand. "
#     subg = "Chips and milk are in Agent's possession"
#     policy = Agent.policy_gen("gpt-4o", act, capt, subg)
#     print("Policy for reaching current goal: ", policy)
    