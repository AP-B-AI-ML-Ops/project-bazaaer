sudo chmod +x kaggle.sh
sed -i 's/\r$//' ./kaggle.sh
./kaggle.sh

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi

cd ./exercise
./servers.sh
