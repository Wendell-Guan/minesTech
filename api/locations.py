"""Vercel serverless function: /api/locations — saved locations CRUD.

Uses /tmp for ephemeral storage on Vercel. For persistent storage,
deploy with server.py on a VPS or add a database.
"""

import json
import os
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler

LOCATIONS_FILE = '/tmp/saved_locations.json'


def read_locations():
    if not os.path.exists(LOCATIONS_FILE):
        return []
    try:
        with open(LOCATIONS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def write_locations(locs):
    with open(LOCATIONS_FILE, 'w') as f:
        json.dump(locs, f, indent=2)


class handler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        self._send_json(read_locations())

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if 'lat' not in body or 'lon' not in body:
            self._send_json({'error': 'lat and lon are required'}, 400)
            return

        locs = read_locations()
        loc = {
            'id': int(time.time() * 1000),
            'name': body.get('name', f'Point {len(locs) + 1}'),
            'lat': float(body['lat']),
            'lon': float(body['lon']),
            'color': body.get('color', '#e74c3c'),
            'savedAt': datetime.utcnow().isoformat() + 'Z'
        }
        locs.append(loc)
        write_locations(locs)
        self._send_json(loc, 201)

    def do_DELETE(self):
        from urllib.parse import urlparse, parse_qs
        query = parse_qs(urlparse(self.path).query)
        loc_id = query.get('id', [None])[0]

        if loc_id:
            locs = read_locations()
            locs = [l for l in locs if str(l.get('id')) != loc_id]
            write_locations(locs)
            self._send_json({'status': 'deleted'})
        else:
            write_locations([])
            self._send_json({'status': 'cleared'})
