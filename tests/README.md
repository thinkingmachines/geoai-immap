
To run test_preprocess.py, download files
```
gsutil -q -m cp gs://immap-gee/gee_itagui_2021-2021.tif data/raw/
```

To run test_predict.py, download files
```
gsutil -q -m cp gs://immap-models/model_LR_30k.sav models/
gsutil -q -m cp gs://immap-models/model_RF_30k.sav models/

gsutil -q -m cp gs://immap-images/20220309/itagui_*.tif data/images/
gsutil -q -m cp gs://immap-indices/20220309/indices_itagui_*.tif data/indices/
```