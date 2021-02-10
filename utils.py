import pandas as pd
import geopandas as gpd
import glob
import os

def returngeo():
    if os.path.exists('./data/geolocations.geojson'):
        gdf = gpd.read_file('./data/geolocations.geojson')
    else:
        # read stations file for Tuscani
        stazioni = pd.read_csv("data/stazioni.csv", sep=';', encoding='unicode_escape')
        stazioni = gpd.GeoDataFrame(stazioni, geometry=gpd.points_from_xy(stazioni['lon'], stazioni['lat']))

        # select locations from stazioni
        target_locs = {'Auser': ['Costanza', 'Salicchi', 'La Tura 2'], 'Luco': ['Podere Casetta'], 'Amiata': ['Bugnano', 'Arubure', 'Ermicillio', 'Galleria Alta'], 'Arno': ['Nave di Rosano'], 'Bilancino': ['Bilancino']}
        target = stazioni[((stazioni.name.isin(target_locs.get('Auser')) | stazioni.name.isin(target_locs.get('Luco'))) & (stazioni.tool == 'freatimetro')) | (stazioni.name.isin(target_locs.get('Arno')) & (stazioni.tool == 'idrometro')) | stazioni.name.isin(target_locs.get('Bilancino'))]    
        stazioni_gdf = gpd.GeoDataFrame({'query': 0, 'place': target['name'], 'place_edit': 0, 'param': 'target', 'feat': ['Bilancino', 'Auser', 'Auser', 'Arno', 'Arno', 'Luco', 'Auser']}, geometry=target['geometry'])

        # use geocode to get other locations
        manual_correction = {'Monte Serra': 'Centro Televisivo Monte Serra', 'Piaggione': 'Piaggione, Lucca', 'Monteporzio': 'Monte Porzio Catone', 
        'Monticiano la Pineta': 'Pinete, Monticiano', 'Monteroni Arbia Biena': "Monteroni d'Arbia", 'Ponte Orgia': 'Orgia', 'Petrignano': 'Petrignano, Perugia',
        'S Piero': 'San Piero a Sieve', 'Le Croci': 'Le Croci, Barberino di Mugello', 'S Agata': "Sant'Agata, Firenze", 'Consuma': 'Passo della Consuma, Arezzo',
        'S Savino': 'Monte San Savino, Arezzo', 'S Fiora': 'Santa Fiora, Grosseto', 'Terni': '05100 Terni', 'Laghetto Verde': 'Abbadia San Salvatore'}

        geocode_gdf = gpd.GeoDataFrame()
        for file in glob.glob('./data/kaggle-original/*.csv'):
                df = pd.read_csv(file)

                feat = file.split('_')[-1].replace('.csv', '')
                if feat == 'Canetto':
                    feat = 'Madonna di Caneto'
                cols = [c for c in set(df.columns) if 'Depth' not in c and 'Flow_Rate' not in c and 'Volume' not in c and 'Lake_Level' not in c and 'Hydrometry' not in c]
                cols = set(cols) - set(['Date'])
                params = [c.split('_')[0] for c in cols]
                places = [c.replace('Rainfall', '').replace('Temperature', '').replace('_', ' ').strip() for c in cols]
                places_edit = list(map(manual_correction.get, places, places)) # replace the manual (google) checks

                locs = ["{}, Italy".format(name) for name in places_edit]
                
                # set up geocoder
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="acea-water-test")

                from geopy.extra.rate_limiter import RateLimiter
                geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

                gdfs = gpd.GeoDataFrame({'query': list(locs), 'place': list(places), 'place_edit': list(places_edit), 'param': list(params), 'target': False})
                gdfs = gdfs.groupby('place').agg({'param': lambda x: 'Both' if len(list(x)) > 1 else x, 'query': 'first'}) # agg. locations with both temperature & rainfall measurements
                gdfs = gdfs.reset_index()

                gdfs['location'] = gdfs['query'].apply(geocode)
                gdfs['adress'] = gdfs['location'].apply(lambda loc: loc.address if loc else None)
                gdfs['feat'] = feat

                from shapely.geometry import Point
                gdfs = gpd.GeoDataFrame(gdfs, geometry=gdfs['location'].apply(lambda loc: Point(loc.longitude, loc.latitude) if loc else None), crs='EPSG:4326')
                gdfs['location'] = gdfs['location'].astype(str)
                geocode_gdf = pd.concat([geocode_gdf, gdfs], axis=0)


        # some final google checks for targets
        locs_manual = {'Petrignano':  [43.104463, 12.533522], 'Doganella': [41.572656, 12.927578], 'Madonna di Canetto': [41.591572, 13.523044], 'Lupa': [42.583980, 12.768410]}
        manual_gdf = gpd.GeoDataFrame({'query': 0, 'place': locs_manual.keys(), 'place_edit': 0, 'param': 'target', 'feat': locs_manual.keys()}, geometry=gpd.points_from_xy([x for y,x in locs_manual.values()], [y for y,x in locs_manual.values()]))

        gdf = pd.concat([stazioni_gdf, geocode_gdf, manual_gdf], axis=0)
        gdf.to_file('./data/geolocations.geojson', driver='GeoJSON')
        return gdf