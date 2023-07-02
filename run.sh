pip3 install pipenv
pipenv --python $(which python3)
pipenv install

pipenv run train -ga 5 -mp 1 -e 1 --device cuda -b 10
