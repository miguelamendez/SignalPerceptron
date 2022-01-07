import os
import sys
full_path = os.path.dirname(os.path.realpath(__file__))
print("config.py",full_path)
sys.path.append(os.path.dirname(os.path.dirname(full_path)))
#print(os.path.dirname(os.path.dirname(full_path)))


