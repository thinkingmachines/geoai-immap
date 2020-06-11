1. Save informal settlement polygon as GeoPackage "area.gpkg"
2. Download satellite images using notebooks/00_Data_Download, (instructions how, inside)
3. Process the images using notebooks/01_Data_Preprocessing

Resulting files and their directories should look like the following:
```
├── data
│   ├── images
│       ├── {area}_2015-2016.tif
│       ├── {area}_2017-2018.tif
│       ├── {area}_2019-2020.tif
│   ├── pos_masks
│       ├── {area}_mask.gpkg
│   ├── admin_bounds
│       ├── {area}_mask.gpkg
│   ├── indices
│       ├── indices_{area}_2015-2016.tif
│       ├── indices_{area}_2017-2018.tif
│       ├── indices_{area}_2019-2020.tif
```

where area is the name of the area you're evaluating for as one word, e.g. Villa del Rosario -> villadelrosario.
