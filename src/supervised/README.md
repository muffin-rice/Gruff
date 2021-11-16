# Usage: 

## 1. 
First download these files and keep the names: 
```
https://console.cloud.google.com/storage/browser/openspiel-data/bridge
```


## 2. 
Run script in desired location with 
```
python train_supervised.py 
--data-dir [DATA_DIR] 
--log-dir [LOG_DIR]
--checkpoint-dir [CHECKPOINT_DIR]...
```

## 3. 

Checkpoints are stored for the model in CHECKPOINT_DIR