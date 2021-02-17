import pandas as pd
import geopandas as gpd
import glob
import os
import umap
import numpy as np
from sklearn.impute import MissingIndicator
import colorlover as cl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# names of all datasets
DATA_SETS = [
    "aquifer_auser",
    "water_spring_amiata",
    "aquifer_petrignano",
    "aquifer_doganella",
    "aquifer_luco",
    "river_arno",
    "lake_bilancino",
    "water_spring_lupa",
    "water_spring_madonna_di_canneto",
]


def preprocess_csvs():
    """
    Preprocess all CSV's to get standardized dataframes
    adding some date related features, and making sure the target column names are renamed to "target_xxxx"
    all casing to lower underscores
    :return: None
    """
    for file in glob.glob("./data/kaggle-original/*.csv"):
        df = pd.read_csv(file, parse_dates=["Date"], dayfirst=True, index_col=["Date"])
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["week"] = df.index.isocalendar().week
        df["day"] = df.index.day
        df["day_of_year"] = df.index.dayofyear

        df = df.sort_index()

        for column in df.columns:
            if column in [
                "Depth_to_Groundwater_SAL",
                "Depth_to_Groundwater_COS",
                "Depth_to_Groundwater_LT2",
                "Flow_Rate_Bugnano",
                "Flow_Rate_Arbure",
                "Flow_Rate_Ermicciolo",
                "Flow_Rate_Galleria_Alta",
                "Depth_to_Groundwater_P24",
                "Depth_to_Groundwater_P25",
                "Depth_to_Groundwater_Pozzo_1",
                "Depth_to_Groundwater_Pozzo_2",
                "Depth_to_Groundwater_Pozzo_3",
                "Depth_to_Groundwater_Pozzo_4",
                "Depth_to_Groundwater_Pozzo_5",
                "Depth_to_Groundwater_Pozzo_6",
                "Depth_to_Groundwater_Pozzo_7",
                "Depth_to_Groundwater_Pozzo_8",
                "Depth_to_Groundwater_Pozzo_9 ",
                "Depth_to_Groundwater_Podere_Casetta",
                "Hydrometry_Nave_di_Rosano",
                "Lake_Level",
                "Flow_Rate",
            ]:
                df.rename(columns={column: "target_{}".format(column)}, inplace=True)
        df.columns = df.columns.str.lower()
        target_name = file.split("/")[-1].split(".")[0]
        df.to_feather("./data/kaggle-preprocessed/{}.feather".format(target_name.lower()))
    return None


def get_geo_df():
    """
    Generates a geolocation dataframe and cache it as json
    Note that this needed quite some manual work and checking to correct for place names that occur multiple times etc
    This would need to be done again for new locations
    :return: gpd.DataFrame
    """
    if os.path.exists('./data/geolocations.geojson'):
        gdf = gpd.read_file('./data/geolocations.geojson')
        return gdf
    else:
        # read stations file for Tuscani
        stazioni = pd.read_csv("data/stazioni.csv", sep=';', encoding='unicode_escape')
        stazioni = gpd.GeoDataFrame(stazioni, geometry=gpd.points_from_xy(stazioni['lon'], stazioni['lat']))

        # select locations from stazioni
        target_locs = {'Auser': ['Costanza', 'Salicchi', 'La Tura 2'], 'Luco': ['Podere Casetta'],
                       'Amiata': ['Bugnano', 'Arubure', 'Ermicillio', 'Galleria Alta'], 'Arno': ['Nave di Rosano'],
                       'Bilancino': ['Bilancino']}
        target = stazioni[((stazioni.name.isin(target_locs.get('Auser')) | stazioni.name.isin(
            target_locs.get('Luco'))) & (stazioni.tool == 'freatimetro')) | (
                                  stazioni.name.isin(target_locs.get('Arno')) & (
                                  stazioni.tool == 'idrometro')) | stazioni.name.isin(
            target_locs.get('Bilancino'))]
        stazioni_gdf = gpd.GeoDataFrame({'query': 0, 'place': target['name'], 'place_edit': 0, 'param': 'target',
                                         'feat': ['Bilancino', 'Auser', 'Auser', 'Arno', 'Arno', 'Luco', 'Auser']},
                                        geometry=target['geometry'])

        # use geocode to get other locations
        manual_correction = {'Monte Serra': 'Centro Televisivo Monte Serra', 'Piaggione': 'Piaggione, Lucca',
                             'Monteporzio': 'Monte Porzio Catone',
                             'Monticiano la Pineta': 'Pinete, Monticiano', 'Monteroni Arbia Biena': "Monteroni d'Arbia",
                             'Ponte Orgia': 'Orgia', 'Petrignano': 'Petrignano, Perugia',
                             'S Piero': 'San Piero a Sieve', 'Le Croci': 'Le Croci, Barberino di Mugello',
                             'S Agata': "Sant'Agata, Firenze", 'Consuma': 'Passo della Consuma, Arezzo',
                             'S Savino': 'Monte San Savino, Arezzo', 'S Fiora': 'Santa Fiora, Grosseto',
                             'Terni': '05100 Terni', 'Laghetto Verde': 'Abbadia San Salvatore'}

        geocode_gdf = gpd.GeoDataFrame()
        for file in glob.glob('./data/kaggle-original/*.csv'):
            df = pd.read_csv(file)

            feat = file.split('_')[-1].replace('.csv', '')
            if feat == 'Canetto':
                feat = 'Madonna di Caneto'
            cols = [c for c in set(df.columns) if
                    'Depth' not in c and 'Flow_Rate' not in c and 'Volume' not in c and 'Lake_Level' not in c and 'Hydrometry' not in c]
            cols = set(cols) - set(['Date'])
            params = [c.split('_')[0] for c in cols]
            places = [c.replace('Rainfall', '').replace('Temperature', '').replace('_', ' ').strip() for c in cols]
            places_edit = list(map(manual_correction.get, places, places))  # replace the manual (google) checks

            locs = ["{}, Italy".format(name) for name in places_edit]

            # set up geocoder
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="acea-water-test")

            from geopy.extra.rate_limiter import RateLimiter
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

            gdfs = gpd.GeoDataFrame(
                {'query': list(locs), 'place': list(places), 'place_edit': list(places_edit), 'param': list(params),
                 'target': False})
            gdfs = gdfs.groupby('place').agg({'param': lambda x: 'Both' if len(list(x)) > 1 else x,
                                              'query': 'first'})  # agg. locations with both temperature & rainfall measurements
            gdfs = gdfs.reset_index()

            gdfs['location'] = gdfs['query'].apply(geocode)
            gdfs['adress'] = gdfs['location'].apply(lambda loc: loc.address if loc else None)
            gdfs['feat'] = feat

            from shapely.geometry import Point
            gdfs = gpd.GeoDataFrame(gdfs, geometry=gdfs['location'].apply(
                lambda loc: Point(loc.longitude, loc.latitude) if loc else None), crs='EPSG:4326')
            gdfs['location'] = gdfs['location'].astype(str)
            geocode_gdf = pd.concat([geocode_gdf, gdfs], axis=0)

        # some final google checks for targets
        locs_manual = {'Petrignano': [43.104463, 12.533522], 'Doganella': [41.572656, 12.927578],
                       'Madonna di Canetto': [41.591572, 13.523044], 'Lupa': [42.583980, 12.768410]}
        manual_gdf = gpd.GeoDataFrame(
            {'query': 0, 'place': locs_manual.keys(), 'place_edit': 0, 'param': 'target', 'feat': locs_manual.keys()},
            geometry=gpd.points_from_xy([x for y, x in locs_manual.values()], [y for y, x in locs_manual.values()]))

        gdf = pd.concat([stazioni_gdf, geocode_gdf, manual_gdf], axis=0)
        gdf.to_file('./data/geolocations.geojson', driver='GeoJSON')
        return gdf


def gather_df(
        dataset_name,
        load_related_data=True,
):
    fname = "./data/kaggle-preprocessed/{}.feather".format(dataset_name)
    if not os.path.exists(fname):
        raise Exception("preprocessed file doesnt exist")
    df = pd.read_feather(fname)
    df = df.set_index(df.index_col)
    df = df.drop("index_col", axis=1).drop("Date", axis=1)
    df['day_of_week'] = pd.to_datetime(df.index).dayofweek
    if load_related_data:
        related_datas = []
        for col in df.columns:
            if "rain" in col:
                location = col.replace("rainfall_", "")
            elif "temperature" in col:
                location = col.replace("temperature_", "")
            else:
                continue
            filename = "./data/nasa-power/{}.feather".format(location)
            if os.path.exists(filename):
                df_related = pd.read_feather(filename)
                df_related = df_related.set_index(df_related.index_col)
                df_related = df_related.drop("index_col", axis=1)
                df_related.columns = [
                    "{}_{}".format(location, c.lower()) for c in df_related.columns
                ]

                related_datas.append(df_related)
            else:
                print("not found: {}".format(col))

        df_related = pd.concat(related_datas)
        df_related = df_related.groupby(df_related.index).max()
        df = pd.merge(df, df_related, how="left", left_index=True, right_index=True)
        for col in df.columns:
            if "_index" in col:
                df = df.drop(col, axis=1)

    df = df[~pd.isna(df.index)]

    for col in df.columns:
        df[col] = df[col].astype(np.float)

    df = df.rename(
        columns={
            "flow_rate_lupa": "target_flow_rate_lupa",
            "depth_to_groundwater_cos": "target_depth_to_groundwater_cos",
            "depth_to_groundwater_pozzo_9": "target_depth_to_groundwater_pozzo_9",
            "flow_rate_madonna_di_canneto": "target_flow_rate_madonna_di_canneto",
        }
    )

    if dataset_name == "aquifer_luco":
        df = df.rename(
            columns={
                "target_depth_to_groundwater_pozzo_1": "depth_to_groundwater_pozzo_1",
                "target_depth_to_groundwater_pozzo_3": "depth_to_groundwater_pozzo_3",
                "target_depth_to_groundwater_pozzo_4": "depth_to_groundwater_pozzo_4",
            }
        )
    for col in [
        "target_depth_to_groundwater_lt2",
        "target_depth_to_groundwater_cos",
        "target_depth_to_groundwater_sal",
        "target_flow_rate_bugnano",
        "target_flow_rate_arbure",
        "target_hydrometry_nave_di_rosano",
        "target_flow_rate_ermicciolo",
        "river_arno",
    ]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df


def prepare_df(df, impute_missing=True, do_extract=True, shift_features=True, location_weights_W=None):
    ignore_cols = ["year", "month", "week", "day", "day_of_year", 'day_of_week']
    
    if impute_missing:
        transformer = MissingIndicator()

        pipeline = transformer
        df_p = pd.DataFrame(pipeline.fit_transform(df))
        df_p.columns = ["missing_{}".format(i) for i in range(df_p.shape[1])]
        df_p.index = df.index

        for col_i in transformer.features_:
            col = df.columns.values[col_i]
            if 'target' in col or 'shift' in col or col in ignore_cols:
                continue

            df[col] = df[col].interpolate(method="quadratic")

    def do_extract(df, columns, location_weights_W, category):
        new_col = (df[columns] * location_weights_W).mean(axis=1).copy()
        df = df.drop(columns, axis=1)
        df[category] = new_col
        return df

    if do_extract:
        for category in [
            "_ts", # earth skin temperature
            "ws10m", # average wind speed at 10m
            "ws10m_min", # min wind speed at 10m
            "ws10m_max",
            "ws50m",
            "ws50m_max",
            "ws50m_min",
            "prectot", # precipitation
            "_ps", # surface pressure
            "qv2m", # specific humidity at 2m
            "rh2m", # relative humidity at 2m
            "t2m", #temperature at 2 meter
            "t2mwet", #wet bulb temp at 2m
            "t2mdew", #dew/frost point at 2m
            "t2m_max",
            "t2m_min",
        ]:
            cols = [c for c in df.columns if c.endswith(category)]
            col_len_categories = len(cols)
            df = do_extract(df, cols ,location_weights_W[:col_len_categories],category)

        cols = [c for c in df.columns if c.startswith('rainfall')]
        col_len_rain = len(cols)
        df = do_extract(df, cols,location_weights_W[col_len_categories:col_len_categories+col_len_rain], category)
        
        cols = [c for c in df.columns if c.startswith('temperature')]
        df = do_extract(df, cols,location_weights_W[col_len_categories+col_len_rain:], category)

    if shift_features:
        for col in df.columns:
            if "shift" not in col and 'missing' not in col and 'target' not in col and col not in ignore_cols:
                for i in range(1, 6, 2):
                    df["{}_shift_{}".format(col, i)] = (
                        df[col].rolling(2).mean().shift(i)
                    )
                for i in range(5, 20, 5):
                    df["{}_shift_{}".format(col, i)] = (
                        df[col].rolling(5).mean().shift(i)
                    )
                for i in range(20, 60, 20):
                    df["{}_shift_{}".format(col, i)] = (
                        df[col].rolling(20).mean().shift(i)
                    )

    

    for col in ['month', 'day_of_week']:
        df[col] = pd.Categorical(df[col])
    df = df.drop(['year', 'month', 'day', 'day_of_year'], axis=1)

    return df

def histograms(df, name, n_cols=3, height=1200):
    colors = cl.scales['12']['qual'].get('Set3')
    numeric_cols = df.select_dtypes('number').columns
    n_rows = -(-len(numeric_cols) // n_cols)  # math.ceil in a fast way, without import
    row_pos, col_pos = 1, 0
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=numeric_cols)

    for i, col in enumerate(numeric_cols):
        # trace extracted from the fig
        trace = go.Histogram(x=df[col].value_counts().index, marker=dict(color=colors[(i+1) % 12]))
        # auto selecting a position of the grid
        if col_pos == n_cols: row_pos += 1
        col_pos = col_pos + 1 if (col_pos < n_cols) else 1
        # adding trace to the grid
        fig.add_trace(trace, row=row_pos, col=col_pos)
    fig.update_layout(template="ggplot2", height=height, title=f'histogram per feature {name}', title_x=0.5, showlegend=False)
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=14)
    
    fig.show()
    
def corr_heatmap(df, name, size=900):
    colors = cl.scales['3']['div']['PRGn']
    
    df_corr = df.corr().where(~np.triu(np.ones(df.corr().shape)).astype(np.bool))
    heat = go.Heatmap(z=df_corr.values,
                      x=df_corr.index,
                      y=df_corr.columns,
                      xgap=1, ygap=1,
                      colorscale=colors,
                      colorbar_thickness=20,
                      colorbar_ticklen=3,
                      hovertext=df.corr().round(2).values,
                      hoverinfo='text'
                       )

    title = f'correlation matrix {name}'               

    layout = go.Layout(template="ggplot2",
                       title_text=title, title_x=0.5, 
                       width=size, height=size,
                       xaxis_showgrid=False,
                       yaxis_showgrid=False,
                       yaxis_autorange='reversed')

    fig=go.Figure(data=[heat], layout=layout)        
    fig.show()
