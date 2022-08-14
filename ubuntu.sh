sudo apt-get update
sudo apt-get install gnome-tweaks gcc g++ clang valgrind binutils python python3 python-pip python3-pip python-tk python3-tk
# sudo apt-get install clang++
sudo pip install numpy scipy matplotlib pandas scikit-learn
sudo pip3 install numpy scipy matplotlib pandas scikit-learn

wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install sublime-text

wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
sudo apt-get update
sudo apt-get install google-chrome-stable

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-390 nvidia-settings
# sudo apt-get install nvidia-driver-418
