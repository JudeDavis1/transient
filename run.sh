pip3 install pipenv
pipenv --python $(which python3)
pipenv install

pipenv run train -ga 5 -mp 1 -e 3 --device cuda -b 10 --lr 0.0001
