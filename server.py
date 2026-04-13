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
from flask import Flask, jsonify, request, send_from_directory
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
    project = os.environ.get('GEE_PROJECT', 'minestech')

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


def build_detection_stack(year):
    """Build the Earth Engine images used by detection and point inspection."""
    # ── 1. Water mask: JRC + coastal + buffered ──
    jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    water_occurrence = jrc.select('occurrence').unmask(-1).rename('water_occurrence')
    # Any pixel ever water >5% of time
    water_mask = water_occurrence.gt(5).rename('water_mask')
    # Keep a smaller exclusion halo so pits next to ponds/rivers can still surface.
    water_buffered = water_mask.focal_max(radius=50, units='meters').rename('water_buffered')

    # Coastal: below 5m elevation
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('elevation')
    coastal_flat = elevation.lt(5).unmask(0).rename('coastal_flat')

    # Combined water mask
    all_water = water_buffered.Or(coastal_flat).rename('all_water')

    # ── 2. Cloud-masked Sentinel-2 composite ──
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filterBounds(get_suriname())
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
          .map(mask_clouds)
          .median())

    # ── 3. Spectral indices ──
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('ndvi')
    ndwi = s2.normalizedDifference(['B3', 'B8']).rename('ndwi')
    mndwi = s2.normalizedDifference(['B3', 'B11']).rename('mndwi')
    bsi = s2.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {'SWIR1': s2.select('B11'), 'RED': s2.select('B4'),
         'NIR': s2.select('B8'), 'BLUE': s2.select('B2')}).rename('bsi')

    # ── 4. Mining detection: aggressive detection ──
    ndvi_rule = ndvi.lt(0.6).rename('ndvi_rule')
    bsi_rule = bsi.gt(-0.2).rename('bsi_rule')
    swir_rule = s2.select('B11').gt(400).rename('swir_rule')
    base_detection = (ndvi_rule.And(bsi_rule).And(swir_rule)
                      .rename('base_detection'))

    # Recover obvious excavations that sit close to water or settling ponds.
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

    # ── 5. Water exclusion (keep it tight) ──
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
        'all_water': all_water,
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


def build_mining_detection(year):
    """
    Detect mining areas in Suriname from Sentinel-2 imagery.

    Uses cloud-masked composites and strict spectral thresholds.
    Excludes all known water bodies (JRC), ocean, rivers, reservoirs.
    Only detects bare soil mining on land.
    """
    stack = build_detection_stack(year)
    return stack['mining'].selfMask().clip(get_suriname())


def _as_bool(value):
    return bool(value) if value is not None else False


def inspect_detection_point(year, lat, lon):
    """Return raw metrics and rule-by-rule decisions for a single point."""
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
        'ndvi_rule': _as_bool(values.get('ndvi_rule')),
        'bsi_rule': _as_bool(values.get('bsi_rule')),
        'swir_rule': _as_bool(values.get('swir_rule')),
        'base_detection': _as_bool(values.get('base_detection')),
        'water_mask': _as_bool(values.get('water_mask')),
        'water_buffered': _as_bool(values.get('water_buffered')),
        'coastal_flat': _as_bool(values.get('coastal_flat')),
        'near_water_excavation': _as_bool(values.get('near_water_excavation')),
        'land_context': _as_bool(values.get('land_context')),
        'ndwi_rule': _as_bool(values.get('ndwi_rule')),
        'mndwi_rule': _as_bool(values.get('mndwi_rule')),
        'nir_gt_green_rule': _as_bool(values.get('nir_gt_green_rule')),
        'detected': _as_bool(values.get('mining'))
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


@app.route('/api/basemap/<int:year>')
def get_basemap(year):
    """Return Sentinel-2 true-color tile URL for a given year (RGB basemap)."""
    if year < 2017 or year > 2026:
        return jsonify({'error': 'Year must be between 2017 and 2026'}), 400

    cache_key = f'basemap_{year}'
    if cache_key in tile_cache:
        cached = tile_cache[cache_key]
        if time.time() - cached['time'] < CACHE_TTL:
            return jsonify({'url': cached['url'], 'year': year})

    try:
        s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterDate(f'{year}-01-01', f'{year}-12-31')
              .filterBounds(get_suriname())
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
              .map(mask_clouds)
              .median())

        rgb = s2.select(['B4', 'B3', 'B2'])
        map_id = rgb.getMapId({
            'min': 300,
            'max': 3500,
            'gamma': 1.3
        })
        tile_url = map_id['tile_fetcher'].url_format

        tile_cache[cache_key] = {'url': tile_url, 'time': time.time()}
        return jsonify({'url': tile_url, 'year': year})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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


@app.route('/api/inspect/<int:year>')
def inspect_point(year):
    """Explain why a point was or was not detected for a given year."""
    if year < 2017 or year > 2026:
        return jsonify({'error': 'Year must be between 2017 and 2026'}), 400

    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    if lat is None or lon is None:
        return jsonify({'error': 'lat and lon query parameters are required'}), 400

    try:
        return jsonify(inspect_detection_point(year, lat, lon))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


LOCATIONS_FILE = os.path.join(DATA_DIR, 'saved_locations.json')
PLOTS_FILE = os.path.join(DATA_DIR, 'saved_plots.json')


def read_locations():
    if not os.path.exists(LOCATIONS_FILE):
        return []
    with open(LOCATIONS_FILE) as f:
        return json.load(f)


def write_locations(locs):
    with open(LOCATIONS_FILE, 'w') as f:
        json.dump(locs, f, indent=2)


def read_plots():
    if not os.path.exists(PLOTS_FILE):
        return []
    with open(PLOTS_FILE) as f:
        return json.load(f)


def write_plots(plots):
    with open(PLOTS_FILE, 'w') as f:
        json.dump(plots, f, indent=2)


@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Return all saved locations."""
    return jsonify(read_locations())


@app.route('/api/locations', methods=['POST'])
def add_location():
    """Add a new saved location."""
    data = request.get_json()
    if not data or 'lat' not in data or 'lon' not in data:
        return jsonify({'error': 'lat and lon are required'}), 400

    locs = read_locations()
    loc = {
        'id': int(time.time() * 1000),
        'name': data.get('name', f'Point {len(locs) + 1}'),
        'lat': float(data['lat']),
        'lon': float(data['lon']),
        'color': data.get('color', '#e74c3c'),
        'savedAt': datetime.utcnow().isoformat() + 'Z'
    }
    locs.append(loc)
    write_locations(locs)
    return jsonify(loc), 201


@app.route('/api/locations/<int:loc_id>', methods=['PATCH'])
def update_location(loc_id):
    """Update name/note of a saved location."""
    data = request.get_json()
    locs = read_locations()
    for loc in locs:
        if loc.get('id') == loc_id:
            if 'name' in data:
                loc['name'] = data['name']
            if 'note' in data:
                loc['note'] = data['note']
            break
    write_locations(locs)
    return jsonify({'status': 'updated'})


@app.route('/api/locations/<int:loc_id>', methods=['DELETE'])
def delete_location(loc_id):
    """Delete a saved location by id."""
    locs = read_locations()
    locs = [l for l in locs if l.get('id') != loc_id]
    write_locations(locs)
    return jsonify({'status': 'deleted'})


@app.route('/api/locations', methods=['DELETE'])
def clear_locations():
    """Delete all saved locations."""
    write_locations([])
    return jsonify({'status': 'cleared'})


@app.route('/api/plots', methods=['GET'])
def get_plots():
    """Return all saved polygons."""
    return jsonify(read_plots())


@app.route('/api/plots', methods=['POST'])
def add_plot():
    """Add a new saved polygon."""
    data = request.get_json()
    if not data or 'coords' not in data:
        return jsonify({'error': 'coords are required'}), 400

    coords = data.get('coords')
    if not isinstance(coords, list) or len(coords) < 4:
        return jsonify({'error': 'coords must contain a closed polygon with at least 4 points'}), 400

    plots = read_plots()
    plot = {
        'id': int(time.time() * 1000),
        'name': data.get('name', f'Area {len(plots) + 1}'),
        'note': data.get('note', ''),
        'color': data.get('color', '#e6a817'),
        'coords': coords,
        'savedAt': datetime.utcnow().isoformat() + 'Z'
    }
    plots.append(plot)
    write_plots(plots)
    return jsonify(plot), 201


@app.route('/api/plots/<int:plot_id>', methods=['PATCH'])
def update_plot(plot_id):
    """Update name/note/color of a saved polygon."""
    data = request.get_json()
    plots = read_plots()
    for plot in plots:
        if plot.get('id') == plot_id:
            if 'name' in data:
                plot['name'] = data['name']
            if 'note' in data:
                plot['note'] = data['note']
            if 'color' in data:
                plot['color'] = data['color']
            break
    write_plots(plots)
    return jsonify({'status': 'updated'})


@app.route('/api/plots/<int:plot_id>', methods=['DELETE'])
def delete_plot(plot_id):
    """Delete a saved polygon by id."""
    plots = read_plots()
    plots = [p for p in plots if p.get('id') != plot_id]
    write_plots(plots)
    return jsonify({'status': 'deleted'})


@app.route('/api/plots', methods=['DELETE'])
def clear_plots():
    """Delete all saved polygons."""
    write_plots([])
    return jsonify({'status': 'cleared'})


@app.route('/api/border')
def get_border():
    """Return Suriname border GeoJSON."""
    border_path = os.path.join(DATA_DIR, 'suriname_border.geojson')
    if not os.path.exists(border_path):
        return jsonify({'error': 'Border data not found'}), 404
    with open(border_path) as f:
        return jsonify(json.load(f))


@app.route('/api/amw/yearly')
def get_amw_yearly():
    """Return yearly AMW mining detection data (2018-2023) for Suriname."""
    path = os.path.join(DATA_DIR, 'amw_suriname_yearly.geojson')
    if not os.path.exists(path):
        return jsonify({'error': 'Yearly AMW data not found'}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route('/api/amw')
def get_amw_data():
    """Return Amazon Mining Watch detection data for Suriname."""
    amw_path = os.path.join(DATA_DIR, 'amw_suriname.geojson')
    if not os.path.exists(amw_path):
        return jsonify({'error': 'AMW data not found'}), 404
    with open(amw_path) as f:
        return jsonify(json.load(f))


@app.route('/api/amw/period/<period>')
def get_amw_period(period):
    """Return AMW single-period GeoJSON (e.g. 2025Q2, 2025Q3, 2025Q4)."""
    path = os.path.join(DATA_DIR, f'amw_period_{period}.geojson')
    if not os.path.exists(path):
        return jsonify({'error': f'Period {period} not found'}), 404
    with open(path) as f:
        return jsonify(json.load(f))


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


@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


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
    port = int(os.environ.get('PORT', 5001))
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
