sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev tk-dev

cd /tmp/
wget https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tgz
tar xzf Python-3.11.11.tgz
cd Python-3.11.11

sudo ./configure --prefix=/opt/python/ --enable-optimizations --with-lto --with-computed-gotos --with-system-ffi --with-openssl=/usr/bin/
# tried --with-openssl= /usr/lib/, /usr/include/
sudo make -j "$(grep -c ^processor /proc/cpuinfo)"
sudo make altinstall
sudo rm /tmp/Python-3.11.11.tgz
