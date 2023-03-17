# -*- coding: utf-8 -*-

import xgboost as xgb
import streamlit as st
import pandas as pd
import geocoder
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
#import geopandas
import folium
from streamlit_folium import st_folium

# Helper functions for transforming address to x,y,z coordinates

def geocoding(input_address): #long first
   g = geocoder.osm(input_address)
   return g.osm['x'], g.osm['y']

def get_cartesian(lat=None,lon=None):
    '''
    Converts latitude and longitude arrays into (x,y,z) coordinates
    Input :
          latitude as array
          longitude as array
    Output :
          x,y,z cartesian coordinates
    '''
    # Change degrees to radians
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 1 # radius of the earth = 6371 km but not needed as we will normalize

    # Convert to cartesian
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)

    return x,y,z

# Building the streamlit

#Loading up the Regression model we created
model = xgb.Booster({'nthread': 2})
model.load_model('xgb_model.model')

#Caching the model for faster loading
@st.cache_data

def predict(horizon, address, zipcode, nbapt1pp, nbapt2pp, nbapt3pp, nbapt4pp, nbapt5pp, sbatapt):

  # turn horizon into the predicted date and extract the detail for date
  time = date.today() + relativedelta(months=+horizon)
  day = time.day
  month = time.month
  year = time.year

  # turn address into (x,y,z) coordinates
  x,y,z = get_cartesian(geocoding(address)[1],geocoding(address)[0])

  # turn zipcode into department number and manually encode it =)
  coddep = zipcode[:2]

  coddep_75 = 0	
  coddep_77	= 0
  coddep_78	= 0
  coddep_91	= 0
  coddep_92	= 0	
  coddep_93	= 0	
  coddep_94	= 0	
  coddep_95	= 0

  if coddep == '75':
    coddep_75 = True
  elif coddep == '77':
    coddep_77 = True
  elif coddep == '78':
    coddep_78 = True
  elif coddep == '91':
    coddep_91 = True  
  elif coddep == '92':
    coddep_92 = True
  elif coddep == '93':
    coddep_93 = True
  elif coddep == '94':
    coddep_94 = True
  else:
    coddep_95 = True

  df = pd.DataFrame([[day, month, year, nbapt1pp, nbapt2pp, nbapt3pp, nbapt4pp, nbapt5pp, sbatapt,coddep_75, coddep_77, coddep_78, coddep_91, coddep_92, coddep_93, coddep_94, coddep_95, x, y, z]],\
          columns=['day', 'month', 'year', 'nbapt1pp', 'nbapt2pp', 'nbapt3pp', 'nbapt4pp', 'nbapt5pp', 'sbatapt', 'coddep_75', 'coddep_77', 'coddep_78', 'coddep_91', 'coddep_92', 'coddep_93', 'coddep_94', 'coddep_95', 'x', 'y', 'z'])
  prediction = model.predict(xgb.DMatrix(df.values))

  return prediction


# Helper functions for calculating the revenue of investing

data = pd.read_excel('cost_computation.xlsx')
data['coddep'].astype(str)

def rent_rev_cost(sbatapt, zipcode, horizon, renov_const):
  '''
  Computes the total costs and potential rental revenues of an appartment after deduction of the construction/renovation time
  Input :
        surface: int,
        zipcode: int,
        horizon: int in months,
        renov_const Renovation or Construction as ast, default to None
  Output :
      total revenues: float  
      total cost: float
        
        
  '''
  total_time_rented = np.amax([(horizon - 48),0])
  dep = int(zipcode[:2])
  
  rent_smeter = total_time_rented * data['price_per_flat'].loc[data['coddep'] == dep].to_numpy()
  print(rent_smeter)
  if renov_const == 'Neither':
    cost = 0
  else:
    cost =  data[renov_const].loc[data['coddep'] == dep].to_numpy()
  total_rev = rent_smeter * sbatapt
  total_cost = (cost+data['price_per_square'].loc[data['coddep'] == dep].to_numpy()) * sbatapt
  return total_rev,total_cost

# building streamlit interface
st.title('The Right Price Predictor')
st.header('Enter your Specifications:')
horizon = st.slider('Predicting Horizon (months):', 0, 120)
address = st.text_input('Address:')
zipcode = st.text_input('Zip Code:')
sbatapt = st.number_input('Total Surface (m²):', min_value=0)   
nbapt1pp = st.number_input('Number of Appartments with 1 room:', min_value=0)
nbapt2pp = st.number_input('Number of Appartments with 2 room:', min_value=0)
nbapt3pp = st.number_input('Number of Appartments with 3 room:', min_value=0)
nbapt4pp = st.number_input('Number of Appartments with 4 room:', min_value=0)
nbapt5pp = st.number_input('Number of Appartments with more than 5 room:', min_value=0)
renov_const = st.selectbox('Will the project require construction or renovation?', ('Construction', 'Renovation','Neither'))

price_xgb = predict(horizon, address, zipcode, nbapt1pp, nbapt2pp, nbapt3pp, nbapt4pp, nbapt5pp, sbatapt)

dept = int(zipcode[:2])
if dept == 75:  
  slope =  3149.3235209600575
elif dept == 77:  
  slope =  354.0234148653532
elif dept== 78 :  
  slope =  1885.3054545099722
elif dept== 91 :  
  slope =  -130.8133202213942
elif dept== 92 :  
  slope =  399.3736597875623
elif dept== 93 :  
  slope =  -338.88537593816545
elif dept== 94 :  
  slope =  1381.8819732699967
else:
  slope =  215.19541104031566

price = price_xgb + slope*horizon
price_unit = price/sbatapt
rent_rev, cost = rent_rev_cost(sbatapt, zipcode, horizon, renov_const)
profit = price + rent_rev - cost

def plot_map(address):
    '''
    Inputs:

    address: str,
    year: int, default =0,
    nb_rooms: int, default = 0,
    price: int, default =0, model returns the price
    surface: int, default=0

    Output:
    Map showing the locations of properties based on the given parameters
    '''
    map = folium.Map(location=(48.868229,2.347402), tiles="OpenStreetMap", zoom_start=4) #center of initial map
    coordinates = [(geocoding(address)[1],geocoding(address)[0])]

    popup = folium.Popup("Address:\n"+str(address)+"<br>" + "<br>" #could be converted from coordinates
                        "Surface (m²): "+str(sbatapt)+"<br>" + "<br>"
                        "Predicted Total Price (€):  "+str(round(price[0],2))+"<br>" + "<br>"
                        "Predicted Unit Price (€/m²):  "+str(round(price_unit[0],2))+"<br>" + "<br>"
                        "Predicted Rental Revenues (€):  "+str(round(rent_rev[0],2))+"<br>" + "<br>"
                        "Predicted Total Cost (€):  "+str(round(-cost[0],2))+"<br>" + "<br>"
                        "Predicted Profit in "+ str(horizon) +" months (€):  "+str(round(profit[0],2))+"<br>" + "<br>", 
                        min_width=300, max_width=300)
    for i in range(len(coordinates)):
        map.add_child(
                folium.Marker(
                    location=coordinates[i],
                    popup= popup))

    sw = min(coordinates)
    ne = max(coordinates)
    map.fit_bounds([sw, ne])
    return map

map  = plot_map(address)
folium_map = st_folium(map, width=800, height=450)