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

def predict(horizon, nbapt1pp, nbapt2pp, nbapt3pp, nbapt4pp, nbapt5pp, sbatapt):

  #import all the zipcodes in IDF
  load_zips = pd.read_csv('communes-IDF-au-01-janvier.csv',sep=";")
  df = load_zips['Geo Point'].str.split(',', expand=True)
    
  # turn address into (x,y,z) coordinates
  df.columns = ['latitude','longitude']
  zone = df[['latitude','longitude']].to_numpy().astype(float)
  x,y,z = get_cartesian(zone[:,0],zone[:,1])
  df['x'] = x
  df['y'] = y
  df['z'] = z
  df= df.drop(['latitude','longitude'], axis=1)

  # turn zipcode into department number and manually encode it =)
  df['coddep'] = load_zips['numdep']
  df['coddep'] = df['coddep'].astype(int)
  df = pd.get_dummies(df, columns=['coddep'])
      
  # turn horizon into the predicted date and extract the detail for date
  time = date.today() + relativedelta(months=+horizon)
  day = time.day
  month = time.month
  year = time.year

  #Fill columns
  df['day'] = day
  df['month'] = month
  df['year'] = year
  df['nbapt1pp'] = nbapt1pp
  df['nbapt2pp'] = nbapt2pp
  df['nbapt3pp'] = nbapt3pp
  df['nbapt4pp'] = nbapt4pp
  df['nbapt5pp'] = nbapt5pp
  df['sbatapt'] = sbatapt
  
  df = df[['day', 'month', 'year', 'nbapt1pp', 'nbapt2pp', 'nbapt3pp', 'nbapt4pp', 'nbapt5pp', 'sbatapt', 'coddep_75', 'coddep_77', 'coddep_78', 'coddep_91', 'coddep_92', 'coddep_93', 'coddep_94', 'coddep_95', 'x', 'y', 'z']]
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
      rental revenues: float  
      total cost: float
        
        
  '''
  total_time_rented = np.amax([(horizon -48),0])
  dep = int(zipcode)
  
  rent_smeter = total_time_rented * data['price_per_flat'].loc[data['coddep'] == dep].to_numpy()
  if renov_const == 'Neither':
    cost = 0
  else:
    cost =  data[renov_const].loc[data['coddep'] == dep].to_numpy()
  total_rev = rent_smeter * sbatapt
  total_cost = (cost+data['price_per_square'].loc[data['coddep'] == dep].to_numpy()) * sbatapt
  return total_rev,total_cost

# building streamlit interface
st.title('The Right Place Predictor')
st.header('Enter your Specifications to identify the 10 most promising locations:')
horizon = st.slider('Predicting Horizon (months):', 0, 120)

sbatapt = st.number_input('Total Surface (m²):', min_value=0)   
nbapt1pp = st.number_input('Number of Appartments with 1 room:', min_value=0)
nbapt2pp = st.number_input('Number of Appartments with 2 room:', min_value=0)
nbapt3pp = st.number_input('Number of Appartments with 3 room:', min_value=0)
nbapt4pp = st.number_input('Number of Appartments with 4 room:', min_value=0)
nbapt5pp = st.number_input('Number of Appartments with more than 5 room:', min_value=0)
renov_const = st.selectbox('Will the project require construction or renovation?', ('Construction', 'Renovation','Neither'))


load_zips = pd.read_csv('communes-IDF-au-01-janvier.csv',sep=";")
zips = load_zips['numdep'] 

price_xgb = predict(horizon, nbapt1pp, nbapt2pp, nbapt3pp, nbapt4pp, nbapt5pp, sbatapt)

price=[]
price_unit=[]
cost=[]
rent_rev=[]
profit=[]
coordinates=[]

for i in range(len(price_xgb)):
    coordinates.append(tuple(load_zips['Geo Point'][i].split(',')))
    dept = int(zips[i])
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
    
    price.append(price_xgb[i] + slope*horizon)
    price_unit.append(price[i]/sbatapt)
    rent_rev_t, cost_t = rent_rev_cost(sbatapt, dept, horizon, renov_const)
    rent_rev.append(rent_rev_t)
    cost.append(cost_t)
    profit.append(float(price[i] + rent_rev_t - cost_t))

def top_k_sort(a, k):
    return np.argsort(a)[-k:]

top10_idx= top_k_sort(np.array(profit),10).flatten()
price = np.array(price)[top10_idx]
price_unit = np.array(price_unit)[top10_idx]
cost = np.array(cost)[top10_idx]
profit = np.array(profit)[top10_idx]
coordinates = [coordinates[i] for i in list(top10_idx)]

def plot_map():
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
    map = folium.Map(location=coordinates[0], tiles="OpenStreetMap", zoom_start=0.001) #center of initial map
    
    popup=[]
    for i in range(len(coordinates)):
        popup.append(folium.Popup("Coordinates:\n"+str(coordinates[i])+"<br>" + "<br>" #could be converted from coordinates
                            "Surface (m²): "+str(sbatapt)+"<br>" + "<br>"
                            "Predicted Total Price (€):  "+str(round(price[i],2))+"<br>" + "<br>"
                            "Predicted Unit Price (€/m²):  "+str(round(price_unit[i],2))+"<br>" + "<br>"
                            "Predicted Rental Revenues (€):  "+str(round(rent_rev[i][0],2))+"<br>" + "<br>"
                            "Predicted Total Cost (€):  "+str(round(cost[i][0],2))+"<br>" + "<br>"
                            "Predicted Profit in "+ str(horizon) +" months (€):  "+str(round(profit[i],2))+"<br>" + "<br>", 
                            min_width=300, max_width=300))
        map.add_child(
                folium.Marker(
                    location=coordinates[i],
                    popup= popup[i]))

    sw = min(coordinates)
    ne = max(coordinates)
    map.fit_bounds([sw, ne])
    return map

map  = plot_map()
folium_map = st_folium(map, width=800, height=450)