# pip3 install pipenv
# pipenv --python $(which python3)
# pipenv install

device=$1

pipenv run train -ga 1 -mp -e 1 --device $device -b 1 --lr 0.0005 -d 0.0 --from-pretrained model_cache
