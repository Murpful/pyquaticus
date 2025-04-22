import os

checkpoint_path = os.path.abspath("/home/matthew/Desktop/couchquaticus/pyquaticus/ray_test/iter_5")
print("Checkpoint path:", checkpoint_path)
if not os.path.exists(checkpoint_path):
    print("Checkpoint directory not found!")
else:
    print("Checkpoint directory exists. Contents:", os.listdir(checkpoint_path))
