'''functions for gee dl and deflatecrop, and dicts for gee settings'''

import math
import ee
ee.Authenticate()
ee.Initialize()

import subprocess
from tqdm import tqdm
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta

# -----
# gee dl
# -----

def imports2(img) :
    s2 = img.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'])\
             .divide(10000).addBands(img.select('QA60'))\
             .set('solar_azimuth',img.get('MEAN_SOLAR_AZIMUTH_ANGLE'))\
             .set('solar_zenith',img.get('MEAN_SOLAR_ZENITH_ANGLE'))
    #['B1','B2','B3','B4','B6','B8A','B9','B10', 'B11','B12'],\
    #['aerosol', 'blue', 'green', 'red', 'red2','red4','h2o', 'cirrus','swir1', 'swir2'])\
    return s2

def rescale(img, thresholds):
    """
    Linear stretch of image between two threshold values.
    """
    return img.subtract(thresholds[0]).divide(thresholds[1] - thresholds[0])

def sentinelCloudScore(img):
    """
    Computes spectral indices of cloudyness and take the minimum of them.

    Each spectral index is fairly lenient because the group minimum 
    is a somewhat stringent comparison policy. side note -> this seems like a job for machine learning :)

    originally written by Matt Hancher for Landsat imagery
    adapted to Sentinel by Chris Hewig and Ian Housman
    """

    # cloud until proven otherwise
    score = ee.Image(1)

    # clouds are reasonably bright
    score = score.min(rescale(img.select(['B2']), [0.1, 0.5]))
    score = score.min(rescale(img.select(['B1']), [0.1, 0.3]))
    #score = score.min(rescale(img.select(['B1']).add(img.select(['B10'])), [0.15, 0.2]))
    score = score.min(rescale(img.select(['B4']).add(img.select(['B3'])).add(img.select('B2')), [0.2, 0.8]))

    # clouds are moist
    ndmi = img.normalizedDifference(['B8A','B11'])
    score=score.min(rescale(ndmi, [-0.1, 0.1]))

    # clouds are not snow.
    ndsi = img.normalizedDifference(['B3', 'B11'])
    score=score.min(rescale(ndsi, [0.8, 0.6])).rename(['cloudScore'])

    return img.addBands(score)

# author: Nick Clinton
def ESAcloud(img) :
    """
    European Space Agency (ESA) clouds from 'QA60', i.e. Quality Assessment band at 60m

    parsed by Nick Clinton
    """

    qa = img.select('QA60')

    # bits 10 and 11 are clouds and cirrus
    cloudBitMask = int(2**10)
    cirrusBitMask = int(2**11)

    # both flags set to zero indicates clear conditions.
    clear = qa.bitwiseAnd(cloudBitMask).eq(0).And(\
           qa.bitwiseAnd(cirrusBitMask).eq(0))

    # clouds is not clear
    cloud = clear.Not().rename(['ESA_clouds'])

    # return the masked and scaled data.
    return cloud


# Author: Gennadii Donchyts
# License: Apache 2.0
def shadowMask(img,cloudMaskType):
    """
    Finds cloud shadows in images

    Originally by Gennadii Donchyts, adapted by Ian Housman
    """

    def potentialShadow(cloudHeight):
        """
        Finds potential shadow areas from array of cloud heights

        returns an image stack (i.e. list of images) 
        """
        cloudHeight = ee.Number(cloudHeight)

        # shadow vector length
        shadowVector = zenith.tan().multiply(cloudHeight)

        # x and y components of shadow vector length
        x = azimuth.cos().multiply(shadowVector).divide(nominalScale).round()
        y = azimuth.sin().multiply(shadowVector).divide(nominalScale).round()

        # affine translation of clouds
        cloudShift = cloudMask.changeProj(cloudMask.projection(), cloudMask.projection().translate(x, y)) # could incorporate shadow stretch?

        return cloudShift

    # select a cloud mask
    cloudMask = img.select(cloudMaskType)

    # make sure it is binary (i.e. apply threshold to cloud score)
    cloudScoreThreshold = 0.5
    cloudMask = cloudMask.gt(cloudScoreThreshold)

    # solar geometry (radians)
    azimuth = ee.Number(img.get('solar_azimuth')).multiply(math.pi).divide(180.0).add(ee.Number(0.5).multiply(math.pi))
    zenith  = ee.Number(0.5).multiply(math.pi ).subtract(ee.Number(img.get('solar_zenith')).multiply(math.pi).divide(180.0))

    # find potential shadow areas based on cloud and solar geometry
    nominalScale = cloudMask.projection().nominalScale()
    cloudHeights = ee.List.sequence(500,4000,500)        
    potentialShadowStack = cloudHeights.map(potentialShadow)
    potentialShadow = ee.ImageCollection.fromImages(potentialShadowStack).max()

    # shadows are not clouds
    potentialShadow = potentialShadow.And(cloudMask.Not())

    # (modified) dark pixel detection 
    darkPixels = img.normalizedDifference(['B3', 'B12']).gt(0.25)

    # shadows are dark
    shadows = potentialShadow.And(darkPixels).rename(['shadows'])

    # might be scope for one last check here. Dark surfaces (e.g. water, basalt, etc.) cause shadow commission errors.
    # perhaps using a NDWI (e.g. B3 and nir)

    return shadows

# Run the cloud masking code
def cloud_mask(img, mask = True):
    s2 = imports2(img)
    cloud = ESAcloud(s2)
    mask = cloud.eq(0)
    return s2.updateMask(mask)

def sen2median(
        BBOX, 
        FILENAME = 'gee_sample',
        year = 2020, 
        min_dt = None, 
        max_dt = None, 
        cloud_pct = 100,
        mask = True,
        PRODUCT = None,
        verbose = 1,
    ):
    '''
    Downloads Sentinel Year Median Aggregate Composite for specified bounding box and year.
    For status, check https://code.earthengine.google.com/
    
    Args
        BBOX (list of 4 floats): bounding box coordinates; x,y left, top, right, bottom
        min_dt, max_dt (str): 'yyyy-mm-dd' min/max date to search images on. If is None, will use year param
        cloud_pct (int): 0 to 100 filter only to images with Cloud pixel percentage less than cloud_pct
        mask (bool): whether to mask out clouds
        PRODUCT (str): can be 'COPERNICUS/S2' or None, whether to force use L1C on all
    
    References
    https://github.com/samsammurphy/cloud-masking-sentinel2
    https://github.com/giswqs/qgis-earthengine-examples
    https://developers.google.com/earth-engine/python_install#syntax
        
    '''

    # set date window
    if (min_dt is None) | (max_dt is None):
        date1 = f'{year}-01-01'
        date2 = f'{year}-12-31'
    else:
        date1 = min_dt
        date2 = max_dt
        
    # select product
    
    if PRODUCT is None:
        if int(date1[0:4]) <= 2017:
            PRODUCT = 'COPERNICUS/S2' # S2 for L1C <=2017 
        else:
            PRODUCT = 'COPERNICUS/S2_SR' # and S2_SR for L2A

    # select region
    region = ee.Geometry.Rectangle(BBOX) # restrict view to bounding box

    #+++++++++++ DISPLAY IMAGE ++++++++++++++++++++++++++

    #obtain the S2 image
    S2 = (ee.ImageCollection(PRODUCT)
     .filterDate(date1, date2)
     .filterBounds(region)
     .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct)))
    
    if verbose > 0:
        print(f'Processing {FILENAME}')
        print(f'using {PRODUCT}')
        print('Filtering to images with cloud cover < {}'.format(cloud_pct))
        if mask:
            print('with mask')
        else:
            print('with no mask')
    
    # Run the cloud masking code
    def cloud_and_shadow_mask(img, mask = mask) :
        s2 = imports2(img)
        s2 = sentinelCloudScore(s2)
        cloud = ESAcloud(s2)
        shadow = shadowMask(s2,'cloudScore')
        mask_ = cloud.Or(shadow).eq(0)
        if mask:
            return s2.updateMask(mask_)
        else:
            return s2
    
    #call the cloud masking functions
    composite = (S2
      .map(cloud_and_shadow_mask)
      .median())

    # # [WIP] Get the following info from the nested dict
    # # id
    # # PRODUCT_ID
    # # CLOUDY_PIXEL_PERCENTAGE
    # temp = S2.getInfo()
    # # source: https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists
    # def gen_dict_extract(key, var):
    #     if hasattr(var,'iteritems'):
    #         for k, v in var.iteritems():
    #             if k == key:
    #                 yield v
    #             if isinstance(v, dict):
    #                 for result in gen_dict_extract(key, v):
    #                     yield result
    #             elif isinstance(v, list):
    #                 for d in v:
    #                     for result in gen_dict_extract(key, d):
    #                         yield result

    # Export task
    task = ee.batch.Export.image.toCloudStorage(
      image= composite.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']),
      description= FILENAME,
      bucket= 'immap-gee',
      maxPixels= 150000000,
      scale= 10,
      region= region,
      crs= 'EPSG:4326'
    )

    task.start()
    print('Task started')
    # task.status()

    
# -----
# deflatecrop
# ----- 

# text = '''

# # set conda env for these commands - took me 3h to figure out
# eval "$(conda shell.bash hook)"
# conda activate ee

# gsutil cp gs://immap-gee/gee_area_year.tif output_dir

# gdal_translate -co COMPRESS=DEFLATE -co TILED=YES output_dirgee_area_year.tif output_dirDEFLATED_gee_area_year.tif

# gdalwarp -co "COMPRESS=DEFLATE" -cutline adm_dirarea.shp -srcnodata -dstnodata output_dirDEFLATED_gee_area_year.tif output_diroutput.tif

# gsutil cp output_diroutput.tif bucketoutput.tif

# #rm output_dirgee_area_year.tif
# #rm output_dirDEFLATED_gee_area_year.tif
# #rm output_diroutput.tif

# '''

# makes arauca work
text = '''

# set conda env for these commands - took me 3h to figure out
eval "$(conda shell.bash hook)"
conda activate ee

gsutil cp gs://immap-gee/gee_area_year.tif output_dir

gdal_translate -co COMPRESS=DEFLATE -co TILED=YES output_dirgee_area_year.tif output_dirDEFLATED_gee_area_year.tif

gdalwarp -cutline adm_dirarea.shp -srcnodata -dstnodata output_dirDEFLATED_gee_area_year.tif output_dirCROPPED_gee_area_year.tif

gdal_translate -co COMPRESS=DEFLATE -co TILED=YES output_dirCROPPED_gee_area_year.tif output_diroutput.tif

gsutil cp output_diroutput.tif bucketoutput.tif

#rm output_dirgee_area_year.tif
#rm output_dirDEFLATED_gee_area_year.tif
#rm output_dirCROPPED_gee_area_year.tif
#rm output_diroutput.tif

'''

def deflatecrop1(raw_filename, output_dir, adm_dir, tmp_dir, bucket, clear_local = True):
    '''
    Same as deflatecrop_all but focused on one image at a time
    
    Args
        output (str): what filename should look like, modified based on if raw_filename is a list
    '''
    if os.path.exists(tmp_dir + "deflatecrop.sh"):
        os.remove(tmp_dir + 'deflatecrop.sh')
        
    logging.basicConfig(filename = tmp_dir + 'deflatecrop.log', filemode='w',level=logging.DEBUG)
    logging.info((datetime.now() + timedelta(hours = 8)).strftime('%Y-%m-%d %H:%M:%S'))
    logging.info('Running for {}'.format(raw_filename))
    
    split_ = (raw_filename
             .replace('0000000000-0000000000', '')
             .replace('0000000000-0000009472', '')
             .split('_'))
    output = split_[1] + '_' + split_[2] + '.tif'
    area = raw_filename.split('_')[1]

    replacement_txt = (text
            .replace('gee_area_year', raw_filename)
            .replace('area.shp', area + '.shp')
            .replace('output_dir', str(Path(output_dir).resolve()) + '/')
            .replace('adm_dir', str(Path(adm_dir).resolve()) + '/')
            .replace('output.tif', output)
            .replace('bucket', bucket)
           )
    if clear_local:
        replacement_txt = replacement_txt.replace('#rm', 'rm')

    f = open(tmp_dir + "deflatecrop.sh", "w")
    f.write(replacement_txt)
    f.close()
    logging.info('Saving the following shell script in ' + tmp_dir + "deflatecrop.sh")
    logging.info(replacement_txt)

    
    logging.info('Running shell script')
    result = subprocess.run('sh ' + tmp_dir + 'deflatecrop.sh', shell = True, stdout=subprocess.PIPE)
    logging.info(result.stdout)
    logging.info('Saved to {}'.format(bucket + output))
    logging.info('Done!')
        
# def deflatecrop_all(
#         raw_filename, output_dir, adm_dir, tmp_dir,
#         bucket = 'gs://immap-gee/'
#     ):
#     '''
#     Allows support for multi-part images
    
#     Args
#         raw_filename (str): in the format gee_area_year
#         output_dir (str): where to put cropped files in the format /path/to/folder/
#         adm_dir (str): where the admin boundary shape files are in the format /path/to/folder/
#     '''

#     if os.path.exists(tmp_dir + "deflatecrop.sh"):
#         os.remove(tmp_dir + 'deflatecrop.sh')
        
#     logging.basicConfig(filename = tmp_dir + 'deflatecrop.log', filemode='w',level=logging.DEBUG)
    
#     if type(raw_filename) == list:
#         logging.info('looping')
#         i = 1
#         for p in raw_filename:
#             p = p.replace('0000000000-0000000000', '').replace('0000000000-0000009472', '')
#             splt = p.split('_')
#             output = splt[1] + '_' + splt[2] + '.tif'
#             # output = '_'.join(splt[0:2]) + str(i) + '_' + splt[2][0:4] + '.tif'
#             deflatecrop1(
#                 raw_filename = p, 
#                 output_dir = output_dir, 
#                 adm_dir = adm_dir, 
#                 tmp_dir = tmp_dir, 
#                 output = output,
#             )
#             i+= 1
#     else:
#         deflatecrop1(
#             raw_filename = raw_filename, 
#             output_dir = output_dir, 
#             adm_dir = adm_dir,
#             tmp_dir = tmp_dir,
#             output = raw_filename + '.tif',
#         )


