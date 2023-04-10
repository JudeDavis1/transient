import os

root_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["PYTHONPATH"] = root_dir

print("ROOT DIR:", root_dir)
