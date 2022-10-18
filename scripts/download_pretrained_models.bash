mkdir weights

for i in {1..6}
do
    mkdir weights/q${i}
    wget https://github.com/kktsubota/universal-dic/releases/download/pre/wacnn-q${i}.pth -O weights/q${i}/model.pth
done
