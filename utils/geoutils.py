import os
from pathlib import Path
import json
import itertools
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio as rio
from rasterio import plot
from rasterio.fill import fillnodata
from rasterio.windows import Window, transform, bounds
from rasterio.merge import merge
from rasterio.plot import show
from rasterio import features
   

def generate_mask(tiff_file, geofile, output_file, plot=False):
    """Generates a segmentation mask for one TIFF image."""
    
    src = rio.open(tiff_file)
    gdf = gpd.read_file(geofile)
    
    if len(gdf) == 0:
        return
    
    shapes = []
    for index, (idx, x) in enumerate(gdf.iterrows()):
        band = 255/((idx/gdf.shape[0])+1)
        gdf_json = json.loads(gpd.GeoDataFrame(x).T.to_json())
        feature = [gdf_json['features'][0]['geometry']][0]
        shapes.append((feature, 1.0))
        
    image = rio.features.rasterize(
        ((g, v) for (g, v) in shapes),
        out_shape=src.shape, 
        transform=src.transform
    )
  
    out_meta = src.meta.copy()  
    out_meta['dtype'] = rio.uint16
    out_meta['count'] = 1
    out_meta['nodata'] = 0
    out_meta['compress'] = 'deflate'
    with rio.open(output_file, 'w', **out_meta) as dst:
        dst.write(image.astype(rio.uint16), indexes=1)

    if plot:
        f, ax = plt.subplots(1, 3, figsize=(15, 15))
        gdf.plot(ax=ax[0])
        rio.plot.show(src, ax=ax[1], adjust=None)
        rio.plot.show(image, ax=ax[2], adjust=None)

        ax[0].set_title('Vector File')
        ax[1].set_title('TIFF')
        ax[2].set_title('Masked')
        plt.show()

def combine_bands(
    image_name,
    band_files,
    sentinel_dir,
    output_dir=None,
    suffix=None,
    plot=False
):
    """ 
    Combines multiple bands and outputs a TIFF 
    with number of dimension = number of bands.
    
    Parameters
    ----------
    image_name : str
        Name of the Sentinel2 image
    band_files: list
        List of file paths for each of the bands to be aggregated
    sentinel_dir : str
        Path to the Sentinel2 directory
    output_dir : str
        Path to output directory
    suffix : str (default None)
        String suffix to be appended to the image filename
    plot : bool (default False)
        Indicated whether to plot the resulting image
        
    Returns
    -------
    numpy array
        Contains the resulting aggregated image
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
        image.write(np.uint16(band.read(1)), index+1)
    image.close()
    
    return image

def aggregate(
    image_files, 
    output_file,
    input_dir,
    cloudmask_dir,
    tmp_dir,
    method='median',  
    grid_blocks=5, 
    plot=False, 
    normalize=False, 
    fill=False
):
    """
    Aggregates raster files according to method.
    
    Parameters
    ----------
    image_files : list 
        List of raster files to aggregate, no suffix 
        e.g. 'S2A_MSIL2A_20200208T152631_N0214_R025_T18PYT_20200208T173911'
    method : str
        Method on how to aggregate image files. Currently only supports median
    output_file : str
        Directory and filename to output aggregated tiff file 
        i.e. '/path/directory/file.tiff'
    grid_blocks : int (default 5)
        Indiciated the number of many times to divide the shape
        e.g. create 10x10 grid from 1 image
    plot : bool
        Indicates whether to display image in window
    normalize : bool
        Indicates where to normalize per raster based on minmax pixel value
    fill : bool
        Indicates whether to interpolate values in blank parts of the raster
        
    Returns
    -------
    None
        Outputs tiff file to output var
    """
    
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # Check if image files are same dimension
    sizes = []
    for i in range(len(image_files)):
        image_file = image_files[i]
        src = rio.open(input_dir+image_file+'.tiff')
        sizes.append(src.shape)

    assert sum([s == sizes[0] for s in sizes]) == len(image_files)

    # Make windows
    src_shape = sizes[0]
    height, width = int(src_shape[0] / grid_blocks), int(src_shape[1] / grid_blocks)
    grid_indices = list(itertools.product(range(grid_blocks), range(grid_blocks)))

    # Get minmax values for band for normalizing later
    band_cnt = src.count
    if normalize == True:
        minmax = []
        for b in range(band_cnt):
            array = src.read(b + 1)
            minmax.append((
                array.min(), array.max()
            ))
    else:
        minmax = []
    
    # Delete temporary files
    if Path(output_file).is_file(): 
        os.remove(output_file)
    p = Path(tmp_dir)
    tmp_files = [str(f) for f in list(p.glob('example*.tiff'))]
    for f in tmp_files:
        os.remove(f)
        
    grid_cnt = len(grid_indices)
    for idx in tqdm(range(len(grid_indices)), total=len(grid_indices)):
        i, j = grid_indices[idx] 
        row_start, row_stop, col_start, col_stop = i*height, (i+1)*height, j*width, (j+1)*width
        w = Window.from_slices((row_start, row_stop), (col_start, col_stop))
        src = rio.open(input_dir + image_files[0] + '.tiff')
        tfm = transform(w, transform = src.transform)

        for image_file in image_files:
            src = rio.open(input_dir + image_file + '.tiff')
            
            cloud_mask_file = cloudmask_dir + image_file + '.tiff'
            if os.path.isfile(cloud_mask_file):
                cloudmask = rio.open(cloud_mask_file) 
            else:
                cloudmask = None

            bands = read_window(src, w, minmax, plot, normalize)
            masked = remove_clouds(cloudmask, bands, w, image_file, width, height, src, tfm, tmp_dir, plot, fill)
        
        mosaic = get_median(image_files, src_shape, w, bands, idx, width, height, src, tfm, tmp_dir, plot)
    
    # Stitch together windows
    stitch(output_file, tmp_dir)
    
    command = "rm -r {}".format(tmp_dir)
    os.system(command)

def read_window(src, w, minmax, plot = False, normalize = True):
    """
    Reads a window in the tiff file.
    
    Parameters
    ----------
    src : rasterReader
        Output of rio.open('file.tiff')
    w : rasterWindow
        Window to read on
    minmax : list of tuples
        Per band min max pixel values
    plot : boolean
        Whether to show image in window
    normalize : bool
        Normalize per raster based on minmax pixel value
        
    Returns
    -------
    bands : list of numpy arrays
        A list containing the the normalizes bands
    """
    
    band_cnt = src.count
        
    def normalize(band, window, minmax, src):
        """Normalizes numpy arrays into scale 0.0 - 1.0"""
        array_min, array_max = minmax[band - 1]
        array = src.read(band, window = window)
        return ((array - array_min)/(array_max - array_min))

    bands = []
    for b in range(band_cnt):
        if normalize == True: 
            array = normalize(b + 1, w, minmax, src)
        else:
            array = np.uint16(src.read(b + 1, window = w))
        bands.append(array)

    if plot == True:
        rgb = np.dstack((bands[0], bands[1], bands[2]))
        plt.imshow(rgb)
        
    return bands

def remove_clouds(
    cloudmask, 
    bands, 
    w, 
    image_file, 
    width, 
    height, 
    src,
    tfm, 
    tmp_dir,
    plot=False, 
    fill=False
):
    """
    Removes clouds from each band.
    
    Parameters
    ----------
    cloudmask : rasterReader
        Output from rio.open('cloudmaskfile.tiff')
    bands : list of numpy arrays
        output from read_window()
    w : rasterWindow
        Window to read on
    image_file : str
        Raster file to remove clouds on, no suffix 
        e.g. 'S2A_MSIL2A_20200208T152631_N0214_R025_T18PYT_20200208T173911'
    width : int
        Width of 1 window
    height : int
        Height of 1 window
    src : rasterReader
        Output from rio.open('file.tiff')
    tfm : raster Affine Transform
        A function that maps pixel locations to spatial positions, wrt to the window
    plot : boolean
        Indicates whether to show image in window
    fill : bool
        Interpolate values in blank parts of the raster
        
    Returns
    -------
    masked : list of numpy arrays
        Band pixel values in image_file where location of clouds are set to 0
    """
    
    if cloudmask is not None:
        # Read cloud mask
        tmp = cloudmask.read(1, window = w)

        # Set to 0 pixels with cloudprobthreshold 2 and 3, and set 1 to everything else
        cloudmask_w = 1 - np.logical_or(tmp == 2, tmp == 3) 
    else:
        cloudmask_w = 1
    
    if plot == True:
        plt.imshow(cloudmask_w == 0)

    ## Clip cloud polygons from window
    masked = []
    for b in range(0, len(bands)):
        array = cloudmask_w * bands[b] # set to 0 in band all that has cloud pixel
        array = np.uint16(array)
        if fill == True: 
            array = fillnodata(array, 1-(array == 0))
        masked.append(array)

    if plot == True:
        rgb = np.dstack((masked[0], masked[1], masked[2]))
        plt.imshow(rgb)

    ## Write to a temp file
    tmp = rio.open(
        tmp_dir + image_file + '_window.tiff', 
        'w',
        driver='Gtiff',
        count=len(bands),
        width=width,
        height=height,
        crs=src.crs,
        dtype=bands[0].dtype,
        transform=tfm,
        compress='deflate',
    )
    for index, band in enumerate(masked):
        tmp.write(band, index+1)
    tmp.close()
    
    return masked

def get_median(image_files, src_shape, w, bands, grid_idx, width, height, src, tfm, tmp_dir, plot = False):
    """
    Get median for tiff files.
    Source: https://automating-gis-processes.github.io/CSC18/lessons/L6/raster-mosaic.html
    
    Parameters
    ----------
    image_files : list of str
        Raster files to aggregate, no suffix 
        e.g. 'S2A_MSIL2A_20200208T152631_N0214_R025_T18PYT_20200208T173911'
    src_shape : tuple of int
        Number of pixel values in height and width of the whole raster file
    w : rasterWindow
        Window to read on
    bands : list of numpy arrays
        Output from read_window()
    grid_idx : int
        Indicates which part of the index currently being processed
    width : int
        Width of 1 window
    height : int
        Height of 1 window
    src : rasterReader
        Output from rio.open('file.tiff')
    tfm : raster Affine Transform
        A function that maps pixel locations to spatial positions, wrt to the window
    plot : boolean
        Whether to show image in window
    
    Returns
    -------
    mosaic : list of numpy arrays
        Band pixel values in image_file where location of clouds are set to 0
    """

    src_files_to_mosaic = []
    for i in range(len(image_files)):
        src_file = rio.open(tmp_dir + image_files[i] + '_window.tiff')
        src_files_to_mosaic.append(src_file)
        
    def median_r(old_data, new_data, old_nodata, new_nodata):
        """Gets the median for a set of rasters"""
        
        mask = np.logical_and(~old_nodata, ~new_nodata)
        old_data[mask] = np.median([old_data[mask], new_data[mask]], axis = 0)

        mask = np.logical_and(old_nodata, ~new_nodata)
        old_data[mask] = new_data[mask]
        
    mosaic, out_trans = merge(
        src_files_to_mosaic, 
        precision = 50,
        nodata = 0, # don't include 0 in median calculation
        method = median_r
    ) 
    
    if plot == True:
        rgb = np.dstack((mosaic[0], mosaic[1], mosaic[2]))
        plt.imshow(rgb)

    ## Write to specific bands in median tif file
    with rio.open(
            tmp_dir + 'example{}.tiff'.format(grid_idx), 
            'w+',
            driver='GTiff', 
            width=width, 
            height=height,
            crs=src.crs,
            transform=tfm,
            count=len(bands),
            dtype=bands[0].dtype,
            compress='deflate',
    ) as dst:
        for i in range(len(bands)):
            dst.write_band(
                i + 1,
                mosaic[i, :, :]
            )
        dst.close()
    return mosaic

def stitch(output_file, tmp_dir):
    """
    Merges all raster files to one
    Source: https://gis.stackexchange.com/questions/230553/merging-all-tiles-from-one-directory-using-gdal
    """
    
    p = Path(tmp_dir)
    file_list = [str(f) for f in list(p.glob('example*.tiff'))]
    files_string = " ".join(file_list)
    command = "gdal_merge.py -o {} -of gtiff ".format(output_file) + files_string
    os.system(command)