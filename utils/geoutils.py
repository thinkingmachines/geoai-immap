import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio as rio
from rasterio import features
import rasterio.mask

def ndvi(b):
    return (b["B8"] - b["B4"]) / (b["B8"] + b["B4"])

def ndbi(b):
    return (b["B11"] - b["B9"]) / (b["B11"] + b["B9"])

def savi(b):
    return 1.5 * (b["B9"] - b["B4"]) / (b["B9"] + b["B4"] + 0.5)

def mndwi(b):
    return (b["B3"] - b["B11"]) / (b["B3"] + b["B11"])

def ui(b):
    return (b["B7"] - b["B5"]) / (b["B7"] + b["B5"])

def nbi(b):
    return b["B4"] * b["B11"] / b["B9"]

def brba(b):
    return b["B4"] / b["B11"]

def nbai(b):
    return (b["B11"] - b["B12"] / b["B3"]) / (b["B11"] + b["B12"] / b["B3"])

def mbi(b):
    return (b["B12"] * b["B4"] - b["B9"] ** 2) / (b["B4"] + b["B9"] + b["B12"])


def baei(b):
    return (b["B4"] + 0.3) / (b["B3"] + b["B11"])


def ibi(b):
    # Source: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1

    # threshold
    t = 0.05

    # normalize to (-1,1)
    ndbi_t, savi_t, mndwi_t =  b["ndbi"], b["savi"], b["mndwi"] #ndbi(), savi(), mndwi()
    ndbi_n = 2 * (ndbi_t - ndbi_t.min()) / (ndbi_t.max() - ndbi_t.min()) - 1
    savi_n = 2 * (savi_t - savi_t.min()) / (savi_t.max() - savi_t.min()) - 1
    mndwi_n = 2 * (mndwi_t - mndwi_t.min()) / (mndwi_t.max() - mndwi_t.min()) - 1

    # remove outliers
    temp = (ndbi_n - (savi_n + mndwi_n) / 2) / (ndbi_n + (savi_n + mndwi_n) / 2)
    vv = pd.DataFrame({"col": temp.reshape(-1, 1)[:, 0]})
    cutoffs = list(vv["col"].quantile([t / 2, 1 - t / 2]))

    temp[temp <= cutoffs[0]] = cutoffs[0]
    temp[temp >= cutoffs[1]] = cutoffs[1]

    return temp

def save_predictions(pred, image_src, output_file):
    """
    Saves the predictions as a TIFF file, based on a source image.
    
    Args:
        pred (numpy array) : The array containing the predictions
        image_src (str) : Path to the source image to be used as a reference file
        
    Returns:
        None 
    """
    
    with rio.open(image_src) as src:
        out_image = np.array(pred).reshape(
            (src.height,src.width)
        )
        out_meta = src.meta

        out_meta.update({
            "driver": "GTiff",
            "height": src.height,
            "width": src.width,
            "count": 1,
            'nodata': -1,
            "dtype": np.float64
        })
        
        with rio.open(output_file, "w", **out_meta, compress='deflate') as dest:
            dest.write(out_image, 1)
    

def read_bands(area_dict, area):
    """
    Reads the bands for each image of each area and calculates 
    the derived indices. 
    
    Args:
        area_dict (dict) : Python dictionary containing the file paths per area
        area (str) : The area of interest (AOI)
        
    Returns:
        data (pd.DataFrame) : The resulting pandas dataframe containing the raw spectral
                              bands and derived indices
    """
    
    data = []
    image_list = area_dict[area]['images_cropped']
    
    # Iterate over each year
    for image_file in tqdm(image_list, total=len(image_list)):
        year = image_file.split('_')[-1].split('.')[0]
        
        # Read each band
        subdata = dict()
        raster = rio.open(image_file)
        for band_idx in range(raster.count):
            band = raster.read(band_idx+1).ravel()
            subdata['B{}'.format(band_idx+1)] = band
        
        # Get derived indices
        subdata["ndvi"] = ndvi(subdata)
        subdata["ndbi"] = ndbi(subdata)
        subdata["savi"] = savi(subdata)
        subdata["mndwi"] = mndwi(subdata)
        subdata["ui"] = ui(subdata)
        subdata["nbi"] = nbi(subdata)
        subdata["brba"] = brba(subdata)
        subdata["nbai"] = nbai(subdata)
        subdata["mbi"] = mbi(subdata)
        subdata["baei"] = baei(subdata)
        
        # Cast to pandas subdataframe
        subdata = pd.DataFrame(subdata).fillna(0)
        subdata.columns = [
            column + '_' + str(year) 
            for column in subdata.columns
        ]
        data.append(subdata)
        del subdata
    
    data = pd.concat(data, axis=1)
    
    return data

def generate_training_data(area_dict):
    """
    Generates training data consisting of pixels. The script obtains the raw spectrals 
    bands and calculates the derived indices for each year (2016-2020) for each area. 
    The resulting dataframe also contains a column containing the target label and 
    a column indicating the area of each pixel. 
    
    Args:
        area_dict (dict) : Python dictionary containing the file paths per area
        
    Returns:
        data (pd.DataFrame) : The resulting pandas dataframe containing the training data
        area_code (dict) : A Python dictionary containing the numerical codes for each area
                           e.g. {'maicao': 0, 'riohacha': 1, 'uribia': 2}
    """
    
    data = []
    area_code = {}

    for idx, area in enumerate(area_dict):
        print('Reading {}...'.format(area))

        # Read positive target mask
        pos_mask = rio.open(area_dict[area]['pos_mask_tiff'])
        pos_mask = pos_mask.read(1).ravel()

        # Read negative mask
        neg_mask = rio.open(area_dict[area]['neg_mask_tiff'])
        neg_mask = neg_mask.read(1).ravel()
 
        # Get sum of postive and negative mask
        mask = pos_mask + neg_mask

        # Read bands
        subdata = read_bands(area_dict, area)
        subdata['target'] = mask
        subdata['area'] = idx
        area_code[area] = idx

        # Get non-zero rows
        subdata = subdata[subdata.values.sum(axis=1) != 0] 
        data.append(subdata)

    # Concatenate all areas
    data = pd.concat(data)
    
    return data, area_code


def get_filepaths(areas, sentinel_dir, pos_mask_dir, neg_mask_dir):
    """
    Returns a dictionary containing the image filepaths for each area.
    
    Args:
        areas (list) : Python list of strings of the areas of interests (AOIs)
                       e.g. ['maicao', 'riohacha', 'uribia']
        
    Returns:
        dict : A Python dictionary 
    """

    area_dict = {area: dict() for area in areas}

    for area in area_dict:
        
        area_dict[area]["pos_mask_gpkg"] = "{}{}_mask.gpkg".format(pos_mask_dir, area)
        area_dict[area]["neg_mask_gpkg"] = "{}{}-samples.gpkg".format(neg_mask_dir, area)
        
        image_files, image_cropped = [], []
        for image_file in os.listdir(sentinel_dir):
            if area in image_file and "DEFLATE" in image_file:
                image_files.append(sentinel_dir + image_file)
            if area in image_file and "CROPPED" in image_file:
                image_cropped.append(sentinel_dir + image_file)
        
        area_dict[area]["images"] = sorted(image_files)
        area_dict[area]["images_cropped"] = sorted(image_cropped)
    
    return area_dict

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
    gdf2 = gs.reset_index().rename(columns={0: "geometry"})
    gdf_out = gdf2.merge(
        gdf.drop("geometry", axis=1), left_on="level_0", right_index=True
    )
    gdf_out = gdf_out.set_index(["level_0", "level_1"]).set_geometry("geometry")
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
    if "class" in gdf.columns:
        unique_classes = sorted(gdf["class"].unique())
        values = {value: x + 2 for x, value in enumerate(unique_classes)}
        values["informal settlement"] = 1
    
    value = 1.0
    shapes = []
    
    for index, (idx, x) in enumerate(gdf.iterrows()):
        band = 255 / ((idx / gdf.shape[0]) + 1)
        if "class" in x:
            value = values[x["class"]]
        gdf_json = json.loads(gpd.GeoDataFrame(x).T.to_json())
        feature = [gdf_json["features"][0]["geometry"]][0]
        shapes.append((feature, value))
    
    image = rio.features.rasterize(
        ((g, v) for (g, v) in shapes), out_shape=src.shape, transform=src.transform
    ).astype(rio.uint16)

    out_meta = src.meta.copy()
    out_meta["dtype"] = rio.uint16
    out_meta["count"] = 1
    out_meta["nodata"] = 0
    out_meta["compress"] = "deflate"

    with rio.open(output_file, "w", **out_meta) as dst:
        dst.write(image, indexes=1)
    
    if plot:
        f, ax = plt.subplots(1, 3, figsize=(15, 15))
        gdf.plot(ax=ax[0])
        rio.plot.show(src, ax=ax[1], adjust=None)
        rio.plot.show(image, ax=ax[2], adjust=None)

        ax[0].set_title("Vector File")
        ax[1].set_title("TIFF")
        ax[2].set_title("Masked")
        plt.show()
        
    return image, values


def get_pos_raster_mask(area_dict):
    """
    Converts positive vector label files (GPKG) to raster masks (TIFF)
    
    Args:
        area_dict (dict) : Python dictionary containing the file paths per area
        
    Returns:
        area_dict (dict) : The input area_dict with a new entry "pos_mask_tiff"
                           containing the file path of the generated TIFF file.
    """

    for area, value in area_dict.items():
        # Get filepaths
        tiff_file = value["images_cropped"][0]
        shape_file = value["pos_mask_gpkg"]
        target_file = shape_file.replace("gpkg", "tiff")

        # Generate masks
        generate_mask(
            tiff_file=tiff_file,
            shape_file=shape_file,
            output_file=target_file,
            plot=False,
        )

        # Set filepath of raster mask in the area dictionary
        area_dict[area]["pos_mask_tiff"] = target_file
        
    return area_dict


def get_neg_raster_mask(area_dict):
    """
    Converts negative vector label files (GPKG) to raster masks (TIFF)
    
    Args:
        area_dict (dict) : Python dictionary containing the file paths per area
        
    Returns:
        area_dict (dict) : The input area_dict with a new entry "neg_mask_tiff"
                           containing the file path of the generated TIFF file.
    """

    for area, value in area_dict.items():

        # Get filepaths
        tiff_file = value["images_cropped"][0]
        shape_file = value["neg_mask_gpkg"]
        target_file = shape_file.replace("gpkg", "tiff")

        if os.path.isfile(shape_file):

            # Read vector file + geopandas cleanup
            gdf = gpd.read_file(shape_file)
            gdf["class"] = gdf["class"].str.lower()
            gdf = gdf[
                (gdf["class"] == "unoccupied land")
                | (gdf["class"] == "formal settlement")
            ]
            shape_file = shape_file.replace("samples", "masks")
            gdf.to_file(shape_file, driver="GPKG")

            # Generate masks
            _, target_dict = generate_mask(
                tiff_file=tiff_file,
                shape_file=shape_file,
                output_file=target_file,
                plot=False,
            )
        # Set filepath of raster mask in the area dictionary
        area_dict[area]["neg_mask_tiff"] = target_file
        
    return area_dict, target_dict
