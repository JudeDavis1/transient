# pip3 install pipenv
# pipenv --python $(which python3)
# pipenv install

device=$1

pipenv run train -ga 5 -mp -e 1 --device $device -b 10 --lr 0.0003 -d 0.0 --from-pretrained model_cache
