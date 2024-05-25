chmod +x ./kaggle.sh
./kaggle.sh

if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi

cd ./exercise
./servers.sh
