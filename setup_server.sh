#!/bin/bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
exec "$SHELL"
sudo apt-get update; sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
pyenv install 3.8.12
pyenv global 3.8.12
sudo apt install python3.8-venv
sudo apt install libpython3.8-dev

cd data_science_project
python3 -m venv env
source env/bin/activate

pip install wheel
pip install -r requirements.txt

sudo ufw enable
sudo ufw allow 22/tcp
deactivate
sudo apt install nginx
sudo ufw allow 'Nginx HTTP'

sudo mv /home/ubuntu/data_science_project/production/app.service  /etc/systemd/system/app.service
sudo systemctl start app
sudo systemctl enable app

sudo mv /home/ubuntu/data_science_project/production/app  /etc/nginx/sites-available/app

sudo ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/
sudo systemctl restart nginx