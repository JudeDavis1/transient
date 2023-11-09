# pip3 install pipenv
# pipenv --python $(which python3)
# pipenv install

pipenv run train -ga 1 -mp 1 -e 1 --device mps -b 42 --lr 0.0005 -d 0.1
