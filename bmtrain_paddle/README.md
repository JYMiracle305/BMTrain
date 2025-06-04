### 1. pip install -r requirement.txt

### 2. 配置nccl
export NCCL_HOME=/root/ncclpath/

### 3. 执行步骤
```bash
 git clone git@github.com:JYMiracle305/BMTrain.git BMTrain_paddle
 cd ./BMTrain_paddle
 pip install .
 cd ./example
 ./command.sh
```