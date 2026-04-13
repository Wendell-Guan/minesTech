"""Vercel serverless function: /api/detect?year=2024"""

import json
import os
import ee
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Initialize EE once per cold start
_initialized = False


def init_ee():
    global _initialized
    if _initialized:
        return

    # Use service account credentials from env var
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


def build_mining_detection(year):
    suriname = ee.Geometry.Rectangle([-58.1, 1.8, -53.9, 6.1])

    # Water mask: JRC + coastal
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    water_mask = jrc.select('occurrence').gt(10)
    srtm = ee.Image('USGS/SRTMGL1_003').select('elevation')
    coastal_flat = srtm.lt(3)
    land_only = water_mask.Or(coastal_flat).Not()

    # Cloud-masked Sentinel-2 composite
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filterBounds(suriname)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
          .map(mask_clouds)
          .median())

    # Spectral indices
    ndvi = s2.normalizedDifference(['B8', 'B4'])
    bsi = s2.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {'SWIR1': s2.select('B11'), 'RED': s2.select('B4'),
         'NIR': s2.select('B8'), 'BLUE': s2.select('B2')})

    # Bare soil mining detection
    mining = (ndvi.lt(0.2)
              .And(bsi.gt(0.15))
              .And(s2.select('B11').gt(1500))
              .And(s2.select('B8').gt(800))
              .And(s2.select('B12').gt(500)))

    mining = mining.And(land_only)

    # Morphological cleaning
    mining_cleaned = (mining
                      .focal_min(radius=30, units='meters')
                      .focal_max(radius=30, units='meters'))

    return mining_cleaned.selfMask().clip(suriname)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            query = parse_qs(urlparse(self.path).query)
            year = int(query.get('year', ['2024'])[0])

            if year < 2017 or year > 2026:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Year must be 2017-2026'}).encode())
                return

            init_ee()
            mining = build_mining_detection(year)
            map_id = mining.getMapId({
                'min': 0, 'max': 1,
                'palette': ['e6a817']
            })
            tile_url = map_id['tile_fetcher'].url_format

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 's-maxage=3600')
            self.end_headers()
            self.wfile.write(json.dumps({
                'url': tile_url,
                'year': year,
                'source': 'sentinel2'
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
