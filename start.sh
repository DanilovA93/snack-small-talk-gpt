#!/usr/bin/zsh
sudo apt-get update
sudo apt install python3-pip
sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
touch ./output.log
chmod 777 ./output.log
chmod 733 ./server.py
nohup python3 ./server.py > ./output.log &
