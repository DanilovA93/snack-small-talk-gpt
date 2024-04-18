#!/usr/bin/zsh
sudo apt-get update
sudo apt install python3-pip
pip3 install -r requirements.txt
sudo touch ./output.log
sudo chmod 777 ./output.log
sudo chmod 733 ./server.py
sudo nohup python3 ./server.py > ./output.log &
