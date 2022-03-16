import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor 💻")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
#ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])
ram = st.selectbox('RAM(in GB)',sorted(df.Ram.unique()))

# weight
weight = st.number_input('Weight of the Laptop(In KG)')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['CPUType'].unique())

clockspeed = st.number_input('CPU Clock Speed In GHz of the Laptop eg:2.4')

hdd = st.selectbox('HDD(in GB)',sorted(df.HDD.unique()))

ssd = st.selectbox('SSD(in GB)',sorted(df.SSD.unique()))

flash = st.selectbox('Flash Storage(in GB)',sorted(df.FlashStorage.unique()))

hybrid= st.selectbox('Hybrid Storage(in GB)',sorted(df.HybridStorage.unique()))



gpu = st.selectbox('GPU',df['Gpu'].unique())

os = st.selectbox('OS',df['OpSys'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,gpu,os,weight,touchscreen,ips,ppi,clockspeed,cpu,ssd,flash,hybrid,hdd])

    query = query.reshape(1,15)
    prediction = int(np.exp(pipe.predict(query)[0]))
    st.write("The predicted price of this configuration is " ,prediction, " INR")