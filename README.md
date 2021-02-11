# acea-water-prediction

# Competition
https://www.kaggle.com/c/acea-water-prediction

# Evaluation
- MAE & RMSE;
- Data visualisations (heatmaps?);
- Discussion on relation of features to target;
- Feature importance;
- Cite public data;

# Other
- Can be day-timescale or month, prediction month ahead;

# Aquifers
Influence of precipitation and evaporation, other aquifers (eg. seepage), drainage levels, depth to groundwater, drainage volumes (pumping, irrigation?).

## Auser (Outputs: Depth_to_Groundwater_SAL, Depth_to_Groundwater_COS, Depth_to_Groundwater_LT2)
- NORTH (SAL, PAG, COS, DIAL) influences SOUTH (LT2);
- NORTH is unconfined, SOUTH is confined;

## Petrignano (Outputs: Depth_to_Groundwater_P24, Depth_to_Groundwater_P25)
- fed by Chiasco river;

## Doganella (Outputs: Depth_to_Groundwater_Pozzo_1 to 9)
- mainly meteoric infiltration;

## Luco (Outputs: Depth_to_Groundwater_Podere_Casetta)
- mainly meteoric infiltration; 

# Waterspring

## Amiata (Outputs: Flow_Rate_Bugnano, Flow_Rate_Arubure, Flow_Rate_Ermicillio, Flow_Rate_Galleria_Alta)
- mainly meteoric infiltration; 

## Madonna di Canneto (Outputs: Flow_Rate_Madonna_di_Canneto)
- 1010 meters ASL;
- fed by catchment of river Melfa;

## Lupa (Outputs: Flow_Rate_Lupa)
- 375 meters ASL;
- river Nera;

# River

## Arno (Outputs: Hydrometry_Nave_di_Rosano)
- torrential;
- fed by lake Bilancino in summer;

# Lake

## Bilancino (Outputs: Lake_Level, Flow_Rate)
- artificial lake;
- used to refill Arno in summer;
