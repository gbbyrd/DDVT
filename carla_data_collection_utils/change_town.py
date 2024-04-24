import argparse
import carla

# get town from arguments
parser = argparse.ArgumentParser()
    
parser.add_argument(
    "--town", 
    action='store',
    type=str, 
    help='Specify which town to change to.')

args = parser.parse_args()

# change the town
client = carla.Client('localhost', 2000)
world = client.load_world(args.town)