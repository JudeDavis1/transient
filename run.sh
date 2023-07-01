pip3 install pipenv
pipenv --python $(which python3)
pipenv install

pipenv run train -ga 1 -mp 1 -e 3 --device cuda -b 40
