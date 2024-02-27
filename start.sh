#!/usr/bin/zsh
sudo touch ./output.log
sudo chmod 777 ./output.log
sudo nohup python3 ./server.py > ./output.log &
