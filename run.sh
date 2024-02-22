# pip3 install pipenv
# pipenv --python $(which python3)
# pipenv install

device=$1

python3 -m src.model.train -ga 2 -mp -e 5 --device $device -b 48 --lr 0.0003 -d 0.2 --from-pretrained model_cache
