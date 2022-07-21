"""Functions for gee download"""

import math
import ee

ee.Authenticate()
ee.Initialize()

# -----
# gee dl
# -----


def imports2(img):
    s2 = (
        img.select(
            [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B9",
                "B11",
                "B12",
            ]
        )
        .divide(10000)
        .addBands(img.select("QA60"))
        .set("solar_azimuth", img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        .set("solar_zenith", img.get("MEAN_SOLAR_ZENITH_ANGLE"))
    )
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
    score = score.min(rescale(img.select(["B2"]), [0.1, 0.5]))
    score = score.min(rescale(img.select(["B1"]), [0.1, 0.3]))
    score = score.min(
        rescale(
            img.select(["B4"]).add(img.select(["B3"])).add(img.select("B2")),
            [0.2, 0.8],
        )
    )

    # clouds are moist
    ndmi = img.normalizedDifference(["B8A", "B11"])
    score = score.min(rescale(ndmi, [-0.1, 0.1]))

    # clouds are not snow.
    ndsi = img.normalizedDifference(["B3", "B11"])
    score = score.min(rescale(ndsi, [0.8, 0.6])).rename(["cloudScore"])

    return img.addBands(score)


# author: Nick Clinton
def ESAcloud(img):
    """
    European Space Agency (ESA) clouds from 'QA60', i.e. Quality Assessment band at 60m

    parsed by Nick Clinton
    """

    qa = img.select("QA60")

    # bits 10 and 11 are clouds and cirrus
    cloudBitMask = int(2**10)
    cirrusBitMask = int(2**11)

    # both flags set to zero indicates clear conditions.
    clear = (
        qa.bitwiseAnd(cloudBitMask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    )

    # clouds is not clear
    cloud = clear.Not().rename(["ESA_clouds"])

    # return the masked and scaled data.
    return cloud


# Author: Gennadii Donchyts
# License: Apache 2.0
def shadowMask(img, cloudMaskType):
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
        cloudShift = cloudMask.changeProj(
            cloudMask.projection(), cloudMask.projection().translate(x, y)
        )  # could incorporate shadow stretch?

        return cloudShift

    # select a cloud mask
    cloudMask = img.select(cloudMaskType)

    # make sure it is binary (i.e. apply threshold to cloud score)
    cloudScoreThreshold = 0.5
    cloudMask = cloudMask.gt(cloudScoreThreshold)

    # solar geometry (radians)
    azimuth = (
        ee.Number(img.get("solar_azimuth"))
        .multiply(math.pi)
        .divide(180.0)
        .add(ee.Number(0.5).multiply(math.pi))
    )
    zenith = (
        ee.Number(0.5)
        .multiply(math.pi)
        .subtract(
            ee.Number(img.get("solar_zenith")).multiply(math.pi).divide(180.0)
        )
    )

    # find potential shadow areas based on cloud and solar geometry
    nominalScale = cloudMask.projection().nominalScale()
    cloudHeights = ee.List.sequence(500, 4000, 500)
    potentialShadowStack = cloudHeights.map(potentialShadow)
    potentialShadow = ee.ImageCollection.fromImages(potentialShadowStack).max()

    # shadows are not clouds
    potentialShadow = potentialShadow.And(cloudMask.Not())

    # (modified) dark pixel detection
    darkPixels = img.normalizedDifference(["B3", "B12"]).gt(0.25)

    # shadows are dark
    shadows = potentialShadow.And(darkPixels).rename(["shadows"])

    # might be scope for one last check here. Dark surfaces (e.g. water, basalt, etc.) cause shadow commission errors.
    # perhaps using a NDWI (e.g. B3 and nir)

    return shadows


# Run the cloud masking code
def cloud_mask(img, mask=True):
    s2 = imports2(img)
    cloud = ESAcloud(s2)
    mask = cloud.eq(0)
    return s2.updateMask(mask)


def sen2median(
    BBOX,
    FILENAME="gee_sample",
    year=2020,
    min_dt=None,
    max_dt=None,
    cloud_pct=100,
    mask=True,
    PRODUCT=None,
    verbose=1,
    wait=False,
):
    """
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

    """

    # set date window
    if (min_dt is None) | (max_dt is None):
        date1 = f"{year}-01-01"
        date2 = f"{year}-12-31"
    else:
        date1 = min_dt
        date2 = max_dt

    # select product

    if PRODUCT is None:
        if int(date1[0:4]) <= 2017:
            PRODUCT = "COPERNICUS/S2"  # S2 for L1C <=2017
        else:
            PRODUCT = "COPERNICUS/S2_SR"  # and S2_SR for L2A

    # select region
    region = ee.Geometry.Rectangle(BBOX)  # restrict view to bounding box

    # +++++++++++ DISPLAY IMAGE ++++++++++++++++++++++++++

    # obtain the S2 image
    S2 = (
        ee.ImageCollection(PRODUCT)
        .filterDate(date1, date2)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )

    if verbose > 0:
        print(f"Processing {FILENAME}")
        print(f"using {PRODUCT}")
        print("Filtering to images with cloud cover < {}".format(cloud_pct))
        if mask:
            print("with mask")
        else:
            print("with no mask")

    # Run the cloud masking code
    def cloud_and_shadow_mask(img, mask=mask):
        s2 = imports2(img)
        s2 = sentinelCloudScore(s2)
        cloud = ESAcloud(s2)
        shadow = shadowMask(s2, "cloudScore")
        mask_ = cloud.Or(shadow).eq(0)
        if mask:
            return s2.updateMask(mask_)
        else:
            return s2

    # call the cloud masking functions
    composite = S2.map(cloud_and_shadow_mask).median()

    # Export task
    task = ee.batch.Export.image.toCloudStorage(
        image=composite.select(
            [
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B9",
                "B11",
                "B12",
            ]
        ),
        description=FILENAME,
        bucket="immap-gee",
        maxPixels=150000000,
        scale=10,
        region=region,
        crs="EPSG:4326",
    )

    task.start()
    print("Task started")
    if wait:
        task.status()

