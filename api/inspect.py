"""Vercel serverless function: /api/inspect?year=2024&lat=5.2&lon=-55.4"""

import json
import os
import ee
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

_initialized = False


def init_ee():
    global _initialized
    if _initialized:
        return

    sa_key = os.environ.get('GEE_SERVICE_ACCOUNT_KEY')
    project = os.environ.get('GEE_PROJECT', 'minestech')

    if sa_key:
        sa_info = json.loads(sa_key)
        credentials = ee.ServiceAccountCredentials(
            sa_info['client_email'],
            key_data=sa_key
        )
        ee.Initialize(credentials=credentials, project=project,
                      opt_url='https://earthengine-highvolume.googleapis.com')
    else:
        ee.Initialize(project=project,
                      opt_url='https://earthengine-highvolume.googleapis.com')

    _initialized = True


def mask_clouds(image):
    scl = image.select('SCL')
    cloud_mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
                  .And(scl.neq(1)).And(scl.neq(2)).And(scl.neq(11)))
    return image.updateMask(cloud_mask)


def build_detection_stack(year):
    suriname = ee.Geometry.Rectangle([-58.1, 1.8, -53.9, 6.1])

    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    water_occurrence = jrc.select('occurrence').unmask(-1).rename('water_occurrence')
    water_mask = water_occurrence.gt(5).rename('water_mask')
    water_buffered = water_mask.focal_max(radius=50, units='meters').rename('water_buffered')
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elevation')
    coastal_flat = elevation.lt(5).unmask(0).rename('coastal_flat')
    all_water = water_buffered.Or(coastal_flat).rename('all_water')

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filterBounds(suriname)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
          .map(mask_clouds)
          .median())

    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('ndvi')
    ndwi = s2.normalizedDifference(['B3', 'B8']).rename('ndwi')
    mndwi = s2.normalizedDifference(['B3', 'B11']).rename('mndwi')
    bsi = s2.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {'SWIR1': s2.select('B11'), 'RED': s2.select('B4'),
         'NIR': s2.select('B8'), 'BLUE': s2.select('B2')}).rename('bsi')

    ndvi_rule = ndvi.lt(0.6).rename('ndvi_rule')
    bsi_rule = bsi.gt(-0.2).rename('bsi_rule')
    swir_rule = s2.select('B11').gt(400).rename('swir_rule')
    base_detection = (ndvi_rule.And(bsi_rule).And(swir_rule)
                      .rename('base_detection'))

    near_water_excavation = (ndvi.lt(0.35)
                             .And(bsi.gt(0.05))
                             .And(s2.select('B11').gt(1000))
                             .And(ndwi.lt(0.0))
                             .And(mndwi.lt(0.1))
                             .And(s2.select('B8').gt(s2.select('B3')))
                             .rename('near_water_excavation'))

    land_context = (all_water.Not()
                    .Or(near_water_excavation
                        .And(water_mask.Not())
                        .And(coastal_flat.Not()))
                    .rename('land_context'))

    ndwi_rule = ndwi.lt(0.05).rename('ndwi_rule')
    mndwi_rule = mndwi.lt(0.15).rename('mndwi_rule')
    nir_gt_green_rule = s2.select('B8').gt(s2.select('B3')).rename('nir_gt_green_rule')
    mining = (base_detection
              .And(land_context)
              .And(ndwi_rule)
              .And(mndwi_rule)
              .And(nir_gt_green_rule)
              .rename('mining'))

    return {
        's2': s2,
        'water_occurrence': water_occurrence,
        'water_mask': water_mask,
        'water_buffered': water_buffered,
        'elevation': elevation,
        'coastal_flat': coastal_flat,
        'ndvi': ndvi,
        'ndwi': ndwi,
        'mndwi': mndwi,
        'bsi': bsi,
        'ndvi_rule': ndvi_rule,
        'bsi_rule': bsi_rule,
        'swir_rule': swir_rule,
        'base_detection': base_detection,
        'near_water_excavation': near_water_excavation,
        'land_context': land_context,
        'ndwi_rule': ndwi_rule,
        'mndwi_rule': mndwi_rule,
        'nir_gt_green_rule': nir_gt_green_rule,
        'mining': mining
    }


def as_bool(value):
    return bool(value) if value is not None else False


def inspect_detection_point(year, lat, lon):
    point = ee.Geometry.Point([lon, lat])
    stack = build_detection_stack(year)

    inspection_image = ee.Image.cat([
        stack['s2'].select(['B2', 'B3', 'B4', 'B8', 'B11', 'SCL']),
        stack['water_occurrence'],
        stack['elevation'],
        stack['ndvi'],
        stack['ndwi'],
        stack['mndwi'],
        stack['bsi'],
        stack['water_mask'],
        stack['water_buffered'],
        stack['coastal_flat'],
        stack['ndvi_rule'],
        stack['bsi_rule'],
        stack['swir_rule'],
        stack['base_detection'],
        stack['near_water_excavation'],
        stack['land_context'],
        stack['ndwi_rule'],
        stack['mndwi_rule'],
        stack['nir_gt_green_rule'],
        stack['mining']
    ])

    values = inspection_image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=10,
        maxPixels=1e8
    ).getInfo()

    if not values or 'B2' not in values:
        return {
            'year': year,
            'lat': lat,
            'lon': lon,
            'detected': False,
            'summary': 'No usable Sentinel-2 observation at this point for the selected year.',
            'blockers': ['no_valid_observation'],
            'metrics': {},
            'rules': {}
        }

    rules = {
        'ndvi_rule': as_bool(values.get('ndvi_rule')),
        'bsi_rule': as_bool(values.get('bsi_rule')),
        'swir_rule': as_bool(values.get('swir_rule')),
        'base_detection': as_bool(values.get('base_detection')),
        'water_mask': as_bool(values.get('water_mask')),
        'water_buffered': as_bool(values.get('water_buffered')),
        'coastal_flat': as_bool(values.get('coastal_flat')),
        'near_water_excavation': as_bool(values.get('near_water_excavation')),
        'land_context': as_bool(values.get('land_context')),
        'ndwi_rule': as_bool(values.get('ndwi_rule')),
        'mndwi_rule': as_bool(values.get('mndwi_rule')),
        'nir_gt_green_rule': as_bool(values.get('nir_gt_green_rule')),
        'detected': as_bool(values.get('mining'))
    }

    blockers = []
    if not rules['ndvi_rule']:
        blockers.append('ndvi_too_high')
    if not rules['bsi_rule']:
        blockers.append('bsi_too_low')
    if not rules['swir_rule']:
        blockers.append('swir_too_low')
    if not rules['land_context']:
        if rules['water_buffered']:
            blockers.append('buffered_water_context')
        if rules['coastal_flat']:
            blockers.append('coastal_lowland_mask')
        if not rules['near_water_excavation']:
            blockers.append('near_water_override_not_triggered')
    if not rules['ndwi_rule']:
        blockers.append('ndwi_looks_like_water')
    if not rules['mndwi_rule']:
        blockers.append('mndwi_looks_like_water')
    if not rules['nir_gt_green_rule']:
        blockers.append('nir_not_greater_than_green')

    if rules['detected']:
        if rules['near_water_excavation'] and rules['water_buffered']:
            summary = 'Detected. The point sits in buffered water context, but the near-water excavation override recovered it.'
        else:
            summary = 'Detected. The point passed the mining thresholds and the land-context filters.'
    else:
        summary = 'Not detected. The point failed: ' + ', '.join(blockers) if blockers else 'Not detected. The point did not pass the final rule stack.'

    metrics = {
        'B2': values.get('B2'),
        'B3': values.get('B3'),
        'B4': values.get('B4'),
        'B8': values.get('B8'),
        'B11': values.get('B11'),
        'SCL': values.get('SCL'),
        'elevation_m': values.get('elevation'),
        'water_occurrence': None if values.get('water_occurrence', -1) < 0 else values.get('water_occurrence'),
        'ndvi': values.get('ndvi'),
        'ndwi': values.get('ndwi'),
        'mndwi': values.get('mndwi'),
        'bsi': values.get('bsi')
    }

    return {
        'year': year,
        'lat': lat,
        'lon': lon,
        'detected': rules['detected'],
        'summary': summary,
        'blockers': blockers,
        'metrics': metrics,
        'rules': rules
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            query = parse_qs(urlparse(self.path).query)
            year = int(query.get('year', ['2024'])[0])
            lat = float(query.get('lat', [None])[0])
            lon = float(query.get('lon', [None])[0])

            if year < 2017 or year > 2026:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Year must be 2017-2026'}).encode())
                return

            if lat is None or lon is None:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'lat and lon are required'}).encode())
                return

            init_ee()
            payload = inspect_detection_point(year, lat, lon)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 's-maxage=60')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
