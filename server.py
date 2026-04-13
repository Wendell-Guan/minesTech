"""
MinesTech GEE Backend
Serves Sentinel-2 based mining classification tiles for Suriname (2024+).
Saves detection results as GeoJSON for persistence.
"""

import os
import json
import time
from datetime import datetime
import ee
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Cache tile URLs (they expire after a few hours)
tile_cache = {}
CACHE_TTL = 3600

# Suriname bounding box (lazy-initialized after ee.Initialize)
SURINAME = None


def get_suriname():
    global SURINAME
    if SURINAME is None:
        SURINAME = ee.Geometry.Rectangle([-58.1, 1.8, -53.9, 6.1])
    return SURINAME


def init_ee():
    """Initialize Earth Engine."""
    project = os.environ.get('GEE_PROJECT', None)

    # Try with project
    if project:
        ee.Initialize(project=project, opt_url='https://earthengine-highvolume.googleapis.com')
        return

    # Try without project (older credentials)
    try:
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    except ee.ee_exception.EEException:
        # Try with default high-volume endpoint without project
        try:
            ee.Initialize()
        except ee.ee_exception.EEException as e:
            if 'no project' in str(e).lower():
                print('\n*** ERROR: No GCP project found. ***')
                print('You need to either:')
                print('  1. Create a GCP project and enable Earth Engine API:')
                print('     https://console.cloud.google.com/projectcreate')
                print('     Then: GEE_PROJECT=your-project-id python3 server.py')
                print('')
                print('  2. Or register for Earth Engine:')
                print('     https://code.earthengine.google.com/register')
                print('')
                print('  3. Or re-authenticate with a project:')
                print('     earthengine authenticate --project=your-project-id')
                raise


def mask_clouds(image):
    """Mask clouds and cloud shadows using Sentinel-2 SCL band."""
    scl = image.select('SCL')
    # SCL values: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=cirrus
    cloud_mask = (scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
                  # Also exclude: 1=saturated, 2=dark/shadow, 11=snow
                  .And(scl.neq(1)).And(scl.neq(2)).And(scl.neq(11)))
    return image.updateMask(cloud_mask)


def build_mining_detection(year):
    """
    Detect mining areas in Suriname from Sentinel-2 imagery.

    Uses cloud-masked composites and strict spectral thresholds.
    Excludes all known water bodies (JRC), ocean, rivers, reservoirs.
    Only detects bare soil mining on land.
    """
    # ── 1. Water mask: JRC Global Surface Water + ocean ──
    # JRC: any pixel that has ever been water in 1984-2021
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    # occurrence > 10 means water appeared >10% of observations
    water_mask = jrc.select('occurrence').gt(10)

    # Also mask ocean: everything below ~2m elevation near coast
    # Use SRTM DEM to identify low-lying coastal areas
    srtm = ee.Image('USGS/SRTMGL1_003').select('elevation')
    coastal_flat = srtm.lt(3)  # Below 3m = likely coastal/tidal

    # Combined: permanent/seasonal water OR coastal flat
    all_water = water_mask.Or(coastal_flat)
    land_only = all_water.Not()

    # ── 2. Cloud-masked Sentinel-2 composite ──
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filterBounds(get_suriname())
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
          .map(mask_clouds)
          .median())

    # ── 3. Spectral indices ──
    ndvi = s2.normalizedDifference(['B8', 'B4'])
    bsi = s2.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {'SWIR1': s2.select('B11'), 'RED': s2.select('B4'),
         'NIR': s2.select('B8'), 'BLUE': s2.select('B2')})

    # ── 4. Mining detection: bare soil only ──
    # Mining in Suriname = deforested bare earth with high SWIR
    mining = (ndvi.lt(0.2)
              .And(bsi.gt(0.15))
              .And(s2.select('B11').gt(1500))
              .And(s2.select('B8').gt(800)))  # NIR must be present (not water)

    # ── 5. Apply masks ──
    mining = mining.And(land_only)  # Remove ALL water bodies

    # Also exclude very low SWIR (water residuals)
    mining = mining.And(s2.select('B12').gt(500))

    # ── 6. Morphological cleaning ──
    mining_cleaned = (mining
                      .focal_min(radius=30, units='meters')
                      .focal_max(radius=30, units='meters'))

    return mining_cleaned.selfMask().clip(get_suriname())


@app.route('/api/detect/<int:year>')
def detect_mining(year):
    """Run mining detection for a year, save result, return tile URL."""
    if year < 2017 or year > 2026:
        return jsonify({'error': 'Year must be between 2017 and 2026'}), 400

    # Check tile cache first
    cache_key = f'detect_{year}'
    if cache_key in tile_cache:
        cached = tile_cache[cache_key]
        if time.time() - cached['time'] < CACHE_TTL:
            return jsonify({
                'url': cached['url'], 'year': year,
                'source': 'sentinel2', 'saved': cached.get('saved', False)
            })

    try:
        mining = build_mining_detection(year)

        # Get tile URL (yellow color for detected mining)
        map_id = mining.getMapId({
            'min': 0, 'max': 1,
            'palette': ['e6a817']  # Yellow for "detected/unconfirmed"
        })
        tile_url = map_id['tile_fetcher'].url_format

        # Save detection metadata
        saved = save_detection_record(year, tile_url)

        tile_cache[cache_key] = {
            'url': tile_url, 'time': time.time(), 'saved': saved
        }

        return jsonify({
            'url': tile_url, 'year': year,
            'source': 'sentinel2', 'saved': saved
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/<int:year>')
def get_saved_data(year):
    """Return saved detection data for a year if available."""
    record_path = os.path.join(DATA_DIR, f'detection_{year}.json')
    if not os.path.exists(record_path):
        return jsonify({'error': 'No saved data'}), 404

    with open(record_path) as f:
        record = json.load(f)

    # Check if the saved tile URL is still fresh (< 2 hours old)
    saved_time = record.get('timestamp', 0)
    if time.time() - saved_time < 7200 and record.get('tile_url'):
        return jsonify({
            'url': record['tile_url'],
            'year': year,
            'saved_at': record.get('saved_at', ''),
            'source': 'sentinel2-saved'
        })

    # Tile expired, need to regenerate
    return jsonify({'error': 'Tile expired, regenerate needed'}), 404


@app.route('/api/export/<int:year>')
def export_geojson(year):
    """
    Export mining detection as GeoJSON vectors and save to data/ folder.
    This converts the raster detection to polygons for permanent storage.
    """
    if year < 2017 or year > 2026:
        return jsonify({'error': 'Year must be between 2017 and 2026'}), 400

    try:
        mining = build_mining_detection(year)

        # Convert raster to vector polygons
        vectors = mining.reduceToVectors(
            geometry=get_suriname(),
            scale=30,
            maxPixels=1e8,
            geometryType='polygon',
            eightConnected=True,
            labelProperty='mining'
        )

        # Get GeoJSON from Earth Engine
        geojson = vectors.getInfo()

        # Add metadata
        geojson['metadata'] = {
            'year': year,
            'source': 'Sentinel-2 SR Harmonized',
            'method': 'Spectral indices (NDVI + BSI + NDWI)',
            'exported_at': datetime.utcnow().isoformat() + 'Z',
            'confidence': 'detected (unconfirmed)'
        }

        # Save to data/ folder
        output_path = os.path.join(DATA_DIR, f'mining_detected_{year}.geojson')
        with open(output_path, 'w') as f:
            json.dump(geojson, f)

        return jsonify({
            'status': 'exported',
            'year': year,
            'file': f'data/mining_detected_{year}.geojson',
            'features': len(geojson.get('features', []))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list')
def list_saved():
    """List all saved detection files."""
    files = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('.geojson'):
            path = os.path.join(DATA_DIR, f)
            stat = os.stat(path)
            files.append({
                'file': f,
                'size_kb': round(stat.st_size / 1024, 1),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    return jsonify({'files': files})


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'service': 'minesTech GEE Backend'})


def save_detection_record(year, tile_url):
    """Save detection record with tile URL and timestamp."""
    record_path = os.path.join(DATA_DIR, f'detection_{year}.json')
    record = {
        'year': year,
        'tile_url': tile_url,
        'timestamp': time.time(),
        'saved_at': datetime.utcnow().isoformat() + 'Z',
        'source': 'Sentinel-2 SR Harmonized'
    }
    try:
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2)
        return True
    except Exception:
        return False


if __name__ == '__main__':
    init_ee()
    port = int(os.environ.get('PORT', 5000))
    print(f"""
    ╔══════════════════════════════════════╗
    ║   MinesTech GEE Backend             ║
    ║   http://localhost:{port}              ║
    ║                                      ║
    ║   /api/detect/<year>  Run detection  ║
    ║   /api/export/<year>  Save GeoJSON   ║
    ║   /api/data/<year>    Get saved data ║
    ║   /api/list           List files     ║
    ╚══════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=port, debug=True)
