// set output params
var PRODUCT = 'COPERNICUS/S2'; // S2 for L1C <=2017 and S2_SR for L2A
var FILENAME = 'gee_riohacha_2015_small';
// var FILENAME = 'glcm_riohacha_2016';

// set date window
var date1 = ee.Date.fromYMD(2015,1,1); 
var date2 = ee.Date.fromYMD(2015,12,31); 

// select region
// var BBOX = [-73.171343876, 10.754102423, -72.158197931, 11.754143685]; //satellite image observed
// var BBOX = [-72.292152, 11.686520, -72.244001, 11.734492]; // uribia urban area
var BBOX = [-72.949333, 11.564208, -72.884616, 11.507526]; // riohacha urban area
// var BBOX = [-72.272737, 11.403564, -72.212240, 11.361955]; // maicao urban area
// var BBOX = [-72.52724612099996, 11.560920839000062, -73.17020892181104, 10.948171764015513]; // riohacha admin boundary
// var BBOX = [-72.65839212899994, 11.534938376940019, -72.15850845943176, 11.080548632000045]; // maicao admin boundary
// var BBOX = [-72.37971307699996, 11.747684544661437, -72.15636466747618, 11.523307245000069]; // uribia admin boundary
// var BBOX = [-72.235049, 11.615007, -72.189534, 11.572060]; // green
// var BBOX = [-72.199376, 11.515926, -72.154220, 11.473979]; // desert

var region = ee.Geometry.Rectangle(BBOX); // restrict view to bounding box;

//display country boundary
Map.centerObject(region, 9); 

//+++++++++++ FUNCTIONS ++++++++++++++++++++++++++
function imports2(img) {
  //var s2 = img.select(['B2','B3','B4','B8','B12'])
  var s2 = img.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'])
                       .divide(10000)
                       .addBands(img.select(['QA60']))
                       .set('solar_azimuth',img.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
                       .set('solar_zenith',img.get('MEAN_SOLAR_ZENITH_ANGLE'))
    return s2;
}

// author: Nick Clinton
function ESAcloud(s2) {
  var qa = s2.select('QA60');
  
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = Math.pow(2, 10);
  var cirrusBitMask = Math.pow(2, 11);
  
  // clear if both flags set to zero.
  var clear = qa.bitwiseAnd(cloudBitMask).eq(0);
  // var clear = qa.bitwiseAnd(cloudBitMask).eq(0).and(
  //           qa.bitwiseAnd(cirrusBitMask).eq(0));
  
  var cloud = clear.eq(0);
  return cloud;
}


// Author: Gennadii Donchyts
// License: Apache 2.0
function shadowMask(s2,cloud){

  // solar geometry (radians)
  var azimuth =ee.Number(s2.get('solar_azimuth')).multiply(Math.PI).divide(180.0).add(ee.Number(0.5).multiply(Math.PI));
  var zenith  =ee.Number(0.5).multiply(Math.PI ).subtract(ee.Number(s2.get('solar_zenith')).multiply(Math.PI).divide(180.0));

  // find where cloud shadows should be based on solar geometry
  var nominalScale = cloud.projection().nominalScale();
  var cloudHeights = ee.List.sequence(200,10000,500);
  var shadows = cloudHeights.map(function(cloudHeight){
    cloudHeight = ee.Number(cloudHeight);
    var shadowVector = zenith.tan().multiply(cloudHeight);
    var x = azimuth.cos().multiply(shadowVector).divide(nominalScale).round();
    var y = azimuth.sin().multiply(shadowVector).divide(nominalScale).round();
    return cloud.changeProj(cloud.projection(), cloud.projection().translate(x, y));
  });
  var potentialShadow = ee.ImageCollection.fromImages(shadows).max();
  
  // shadows are not clouds
  potentialShadow = potentialShadow.and(cloud.not());
  
  // (modified by Sam Murphy) dark pixel detection 
  var darkPixels = s2.normalizedDifference(['B3','B12']).gt(0.25).rename(['dark_pixels']);
  
  // shadows are dark
  var shadow = potentialShadow.and(darkPixels).rename('shadows');
  
  return shadow;
}

// Run the cloud masking code
function cloud_and_shadow_mask(img) {
  var s2 = imports2(img);
  var cloud = ESAcloud(s2);
  var shadow = shadowMask(s2,cloud);
  var mask = cloud.or(shadow).eq(0);
  
  return s2.updateMask(mask);
}

// Run the cloud masking code
function cloud_mask(img) {
  var s2 = imports2(img);
  var cloud = ESAcloud(s2);
  var mask = cloud.eq(0);
  return s2.updateMask(mask);
}

//***********************************************************************************
// obtain SENTINEL-2 image and display
//***********************************************************************************

//set vizualization parameters
var vizParams = {'min': 0,'max': [0.2], 'bands':['B4', 'B3', 'B2'] };   //B4, B3, B2

//obtain the S2 image
var S2 = ee.ImageCollection(PRODUCT)
 .filterDate(date1, date2)
 .filterBounds(region)
// .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
 ;

print("S2: ", S2);
//Map.addLayer(S2, vizParams,'S2 initial image');

//call the cloud masking functions
var composite = S2
  .map(cloud_and_shadow_mask)
  .median();

print("composite: ", composite);

// Select the red, green and blue bands
Map.addLayer(composite.clip(region), {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.2},
            'Sentinel-2 RGB',true);

// // Calculate texture using B8 - NIR
// // source: http://www.diva-portal.org/smash/get/diva2:1261937/FULLTEXT01.pdf
// var glcm_input = composite.clip(region).select(['B8'])
//                     .divide(6.56)
//                     .multiply(255)
//                     .toInt();
// print("glcm_input: ", glcm_input);
// var glcm = glcm_input.glcmTexture({size: 4});
// print("glcm: ", glcm);
// Map.addLayer(glcm, {bands: ['B8_asm']},
//             'Texture ASM');
            
// Export.image.toDrive({
//   image: composite.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']),
//   // image: glcm,
//   description: FILENAME,
//   maxPixels: 150000000,
//   scale: 10,
//   region: region
// });

// Export.image.toCloudStorage({
//   image: composite.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']),
//   // image: glcm,
//   description: FILENAME,
//   bucket: 'immap-gee',
//   maxPixels: 150000000,
//   scale: 10,
//   region: region,
//   crs: 'EPSG:4326'
// });
/******************************************************************************************************/ 