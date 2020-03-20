import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm

import rasterio as rio
from rasterio import plot
from rasterio.windows import Window
from rasterio.merge import merge

def combine_bands(
    image_name,
    band_files,
    sentinel_dir,
    output_dir=None,
    suffix=None,
    plot=True
):
    """ Combines multiple bands and outputs a TIFF 
    with number of dimension = number of bands.
    
    Parameters
    ----------
    image_name 
    
    Results
    -------
    """
    
    if output_dir is None:
        output_dir = sentinel_dir
    
    # Read bands
    bands = []
    for band_file in band_files:
        band = rio.open(band_file, driver='JP2OpenJPEG') 
        bands.append(band)
    
    if plot is True:
        fig, ax = plt.subplots(1, len(bands), figsize=(12, 4))
        cmaps = ['Reds', 'Greens', 'Blues']
        for index, (band, cmap) in enumerate(zip(bands, cmaps[:len(bands)])):
            rio.plot.show(band, ax=ax[index], cmap=cmap)
        fig.tight_layout()
    
    # Define filename
    if suffix is None:
        image_file = '{}{}.tiff'.format(
            output_dir, image_name
        )
    else:
        image_file = '{}{}_{}.tiff'.format(
            output_dir, image_name, suffix
        )
        
    # Create TIFF File
    image = rio.open(
        image_file, 
        'w',
        driver='Gtiff',
        count=len(bands),
        width=bands[0].width,
        height=bands[0].height,
        crs=bands[0].crs,
        dtype=np.uint16,
        transform=bands[0].transform,
    )
    for index, band in enumerate(bands):
        image.write(band.read(1).astype(np.uint16), index+1)
    image.close()
    
    return image

def normalize(band, window, minmax, src):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    
    array_min, array_max = minmax[band - 1]
    array = src.read(band, window = window)
    return ((array - array_min)/(array_max - array_min))

def median_r(old_data, new_data, old_nodata, new_nodata):
    mask = np.logical_and(~old_nodata, ~new_nodata)
    old_data[mask] = np.median([old_data[mask], new_data[mask]], axis = 0)

    mask = np.logical_and(old_nodata, ~new_nodata)
    old_data[mask] = new_data[mask]
    
def read_window(src, w):
    # Get minmax values for band for normalizing later
    band_cnt = src.count
    minmax = []
    for b in range(band_cnt):
        array = src.read(b + 1)
        minmax.append((
            array.min(), array.max()
        ))

    # Read 1 window all band for each tif file
    bands = []
    for b in range(band_cnt):
        # array = src.read(b + 1, window = w)
        array = normalize(b + 1, w, minmax, src)
        bands.append(array)

    # Display all bands from 1 window
    # rgb = np.dstack((bands[0], bands[1], bands[2]))
    # plt.imshow(rgb)
    
    return bands

def remove_clouds(cloudmask, bands, w, image_file, tmp_dir, width, height, src):
    # Read cloud mask
    tmp = cloudmask.read(1, window = w)
    
    # Set to 0 pixels with cloudprobthreshold 2 and 3, and set 1 to everything else
    cloudmask_w = 1 - np.logical_or(tmp == 2, tmp == 3) 

    ## Clip cloud polygons from window
    masked = []
    for b in range(0, len(bands)):
        array = cloudmask_w * bands[b] # set to 0 in band all that has cloud pixel
        masked.append(array)

    # rgb = np.dstack((masked[0], masked[1], masked[2]))
    # plt.imshow(rgb)
    
    ## Write to a temp file
    filename = tmp_dir + image_file + '_window.tiff'
    tmp = rio.open(
        filename, 
        'w',
        driver='Gtiff',
        count=len(bands),
        width=width,
        height=height,
        crs=src.crs,
        dtype=bands[0].dtype,
        transform=src.transform,
    )
    for index, band in enumerate(masked):
        tmp.write(band, index+1)
    tmp.close()
    
    return masked

def get_median(
    image_files, 
    src_shape, 
    w, 
    bands, 
    path, 
    src, 
    plot=False
):
    
    ## Get median for tif files
    # source: https://automating-gis-processes.github.io/CSC18/lessons/L6/raster-mosaic.html
    src_files_to_mosaic = []
    for i in range(len(image_files)):
        src_files_to_mosaic.append(rio.open(path + image_files[i] + '_window.tiff'))

    mosaic, out_trans = merge(src_files_to_mosaic, 
                              nodata = 0, # don't include 0 in median calculation
                              method = median_r) # replace with median
    if plot is True:
        rgb = np.dstack((mosaic[0], mosaic[1], mosaic[2]))
        plt.imshow(rgb)
        plt.show()
        plt.close()

    ## Write to specific bands in median tif file
    with rio.open(
        path + 'example.tiff', 
        'w',
        driver='GTiff', 
        width=src_shape[1], 
        height=src_shape[0], 
        crs=src.crs,
        count=len(bands),
        dtype=bands[0].dtype
    ) as dst:
        for i in range(len(bands)):
            dst.write(
                mosaic[i, :, :], 
                window=w, 
                indexes=i + 1
            )
            
    return mosaic

def aggregate(
    image_files, 
    input_dir, 
    cloudmask_dir, 
    tmp_dir, 
    method='median', 
    plot=False
):

    # Check if image files are same dimension
    sizes = []
    for i in range(len(image_files)):
        image_file = image_files[i]
        src = rio.open(input_dir + image_file + '.tiff')
        sizes.append(src.shape)

    assert sum([s == sizes[0] for s in sizes]) == len(image_files)

    # Make windows
    src_shape = sizes[0]
    grid_blocks = 10 # How many times to divide the shape, e.g. create 10x10 grid from 1 image
    height = int(src_shape[0] / grid_blocks)
    width = int(src_shape[1] / grid_blocks)
    grid_indices = list(itertools.product(range(grid_blocks), range(grid_blocks)))

    for idx in tqdm(grid_indices, total=len(grid_indices)):
        i, j = idx 
        row_start = i*height
        row_stop = (i+1)*height
        col_start = j*width
        col_stop = (j+1)*width
        
        w = Window.from_slices(
            (row_start, row_stop), 
            (col_start, col_stop)
        )

        # For each file
        for image_file in image_files:
            src = rio.open(input_dir + image_file + '.tiff')
            cloudmask = rio.open(cloudmask_dir + image_file + '.tiff') 

            bands = read_window(src, w)
            masked = remove_clouds(
                cloudmask, bands, w, image_file, tmp_dir, width, height, src
            )
        
        mosaic = get_median(image_files, src_shape, w, bands, tmp_dir, src, plot=plot)
        
    return mosaic