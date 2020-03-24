import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio as rio
from rasterio import features
import rasterio.mask

def explode(gdf):
    """ 
    Explodes a geodataframe 
    Source: https://gist.github.com/mhweber/cf36bb4e09df9deee5eb54dc6be74d26
    
    Will explode muti-part geometries into single geometries. 
    
    Args:
        gdf (gpd.GeoDataFrame) : Input geodataframe with multi-geometries
        
    Returns:
        gdf (gpd.GeoDataFrame) : Exploded geodataframe with a new index 
                                 and two new columns: level_0 and level_1
                                 
    """
    gs = gdf.explode()
    gdf2 = gs.reset_index().rename(columns={0: 'geometry'})
    gdf_out = gdf2.merge(gdf.drop('geometry', axis=1), left_on='level_0', right_index=True)
    gdf_out = gdf_out.set_index(['level_0', 'level_1']).set_geometry('geometry')
    gdf_out.crs = gdf.crs
    return gdf_out

def generate_mask(tiff_file, shape_file, output_file, plot=False):
    """
    Generates a segmentation mask for one TIFF image.
    
    Args:
        tiff_file (str) : Path to reference TIFF file
        shape_file (str) : Path to shapefile
        output_file (str) : Path to output file 
        
    Returns:
        image (np.array) : A binary mask as a numpy array 
    
    """
    
    src = rio.open(tiff_file)
    gdf = gpd.read_file(shape_file)
    
    values = {}
    if 'class' in gdf.columns:
        values = {value:x+2 for x, value in enumerate(gdf['class'].unique())}
        
    shapes = []
    for index, (idx, x) in enumerate(gdf.iterrows()):
        band = 255/((idx/gdf.shape[0])+1)
        value = 1.0
        if 'class' in x: value = values[x['class']]
        gdf_json = json.loads(gpd.GeoDataFrame(x).T.to_json())
        feature = [gdf_json['features'][0]['geometry']][0]
        shapes.append((feature, value))
        
    image = rio.features.rasterize(
        ((g, v) for (g, v) in shapes),
        out_shape=src.shape, 
        transform=src.transform
    ).astype(rio.uint16)
  
    out_meta = src.meta.copy()  
    out_meta['dtype'] = rio.uint16
    out_meta['count'] = 1
    out_meta['nodata'] = 0
    out_meta['compress'] = 'deflate'
    
    with rio.open(output_file, 'w', **out_meta) as dst:
        dst.write(image, indexes=1)

    if plot:
        f, ax = plt.subplots(1, 3, figsize=(15, 15))
        gdf.plot(ax=ax[0])
        rio.plot.show(src, ax=ax[1], adjust=None)
        rio.plot.show(image, ax=ax[2], adjust=None)

        ax[0].set_title('Vector File')
        ax[1].set_title('TIFF')
        ax[2].set_title('Masked')
        plt.show()
        
    return image