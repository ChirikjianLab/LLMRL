# Generate video for a program. Make sure you have the executable open
import sys
import json

sys.path.append('../simulation')
from unity_simulator.comm_unity import UnityCommunication

# Virtualhome-v1
script = ['<char0> [Walk] <kitchen> (1)', 
          '<char0> [Walk] <pancake> (1)', 
          '<char0> [Grab] <pancake> (1)', 
          '<char0> [Walk] <microwave> (1)',
          '<char0> [Open] <microwave> (1)',
          '<char0> [Putin] <pancake> (1) <microwave> (1)',
          '<char0> [Close] <microwave> (1)'] # Add here your script

# Virtualhome-v1
script = ['<char0> [Walk] <kitchen> (11)', 
          '<char0> [Walk] <chips> (61)', 
          '<char0> [Grab] <chips> (61)', 
          '<char0> [Walk] <milk> (46)',
          '<char0> [Grab] <milk> (46)',
          '<char0> [Walk] <livingroom> (267)',
          '<char0> [Walk] <coffeetable> (268)',
          '<char0> [Putback] <milk> (46) <coffeetable> (268)',
          '<char0> [Walk] <TV> (297)',
          '<char0> [Switchon] <TV> (297)',
          '<char0> [Walk] <livingroom> (267)',
          '<char0> [Walk] <sofa> (276)',
          '<char0> [Sit] <sofa> (276)'
          ] # Add here your script

print('Starting Unity...')
comm = UnityCommunication()

print('Starting scene...')

comm.reset(4)
# _, graph = comm.get_visible_objects
# print(graph)

comm.add_character('Chars/male1')

print('Generating video...')
comm.render_script(script, recording=True, find_solution=True, camera_mode=['PERSON_FROM_BACK'])

print('Generated, find video in simulation/unity_simulator/output/')
