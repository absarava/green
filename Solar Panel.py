#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import datetime as dt
import matplotlib.dates as mdates

#train-test split of data

def train_test_split(df,panel_id_col,split_pct):

    # split into train test sets
    assert split_pct <= 1, 'Check percentage again'
    train_indices = []
    test_indices = []
    for panel_id in df[panel_id_col].unique():
        panel_id_index = df.loc[df[panel_id_col] == panel_id].index
        train = int(split_pct * len(panel_id_index))
        train_indices.append(panel_id_index[0:train])
        test_indices.append(panel_id_index[train:len(panel_id_index)])
    train_indices = [item for sublist in train_indices for item in sublist]
    test_indices = [item for sublist in test_indices for item in sublist]

    train_data = df.loc[train_indices].reset_index(drop=True)
    test_data = df.loc[test_indices].reset_index(drop=True)
    return train_data, test_data

pg1 = pd.read_csv("Plant_1_Generation_Data.csv")
pg2 = pd.read_csv("Plant_2_Generation_Data.csv")
ws1 = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")
ws2 = pd.read_csv("Plant_2_Weather_Sensor_Data.csv")

pg1.drop('PLANT_ID',1,inplace=True)
ws1.drop('PLANT_ID',1,inplace=True)
pg1['DATE_TIME']= pd.to_datetime(pg1['DATE_TIME'],format='%d-%m-%Y %H:%M')
ws1['DATE_TIME']= pd.to_datetime(ws1['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')


#---------Explaratory Data Analysis----------

# Irradiation vs. Temp (remove where irradiation = 0)
df_train_plant = plant2[0]

# df_ws1 = df_ws1[df_ws1['IRRADIATION']!= 0]

# Daily and Total Yield

df_gen1=pg1.groupby('DATE_TIME').sum().reset_index()
df_gen1['time']=df_gen1['DATE_TIME'].dt.time

fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))
# daily yield plot
df_gen1.plot(x='DATE_TIME',y='DAILY_YIELD',color='navy',ax=ax[0])
# AC & DC power plot
df_gen1.set_index('time').drop('DATE_TIME',1)[['AC_POWER','DC_POWER']].plot(style='o',ax=ax[1])

ax[0].set_title('Daily yield',)
ax[1].set_title('AC power & DC power during day hours')
ax[0].set_ylabel('kW',color='navy',fontsize=17)
plt.show()


# Irradiation,Ambient and Module temperature

df_sens=df_gen1.groupby('DATE_TIME').sum().reset_index()
df_sens['time']=df_sens['DATE_TIME'].dt.time

fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))
# daily yield plot
df_sens.plot(x='time',y='IRRADIATION',ax=ax[0],style='o')
# AC & DC power plot
df_sens.set_index('DATE_TIME').drop('time',1)[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].plot(ax=ax[1])

ax[0].set_title('Irradiation during day hours',)
ax[1].set_title('Ambient and Module temperature')
ax[0].set_ylabel('W/m',color='navy',fontsize=17)
ax[1].set_ylabel('°C',color='navy',fontsize=17)

plt.show()


## Real DC power converted
#### As we can see below PLANT_1 it's able to convert about only 9-10 % of DC POWER produced by module: Inverters are inefficient!


losses=pg1.copy()
losses['day']=losses['DATE_TIME'].dt.date
losses=losses.groupby('day').sum()
losses['losses']=losses['AC_POWER']/losses['DC_POWER']*100

losses['losses'].plot(style='o--',figsize=(17,5),label='Real Power')

plt.title('% of DC power converted in AC power',size=17)
plt.ylabel('DC power converted (%)',fontsize=14,color='red')
plt.axhline(losses['losses'].mean(),linestyle='--',color='gray',label='mean')
plt.legend()
plt.show()


## DC power generated during day hours
sources=pg1.copy()
sources['time']=sources['DATE_TIME'].dt.time
sources.set_index('time').groupby('SOURCE_KEY')['DC_POWER'].plot(style='o',legend=True,figsize=(20,10))
plt.title('DC Power during day for all sources',size=17)
plt.ylabel('DC POWER ( kW )',color='navy',fontsize=17)
plt.show()

## Which inverter is underperforming?

dc_gen=pg1.copy()
dc_gen['time']=dc_gen['DATE_TIME'].dt.time
dc_gen=dc_gen.groupby(['time','SOURCE_KEY'])['DC_POWER'].mean().unstack()

cmap = sns.color_palette("Spectral", n_colors=12)

fig,ax=plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,6))
dc_gen.iloc[:,0:11].plot(ax=ax[0],color=cmap)
dc_gen.iloc[:,11:22].plot(ax=ax[1],color=cmap)

ax[0].set_title('First 11 sources')
ax[0].set_ylabel('DC POWER ( kW )',fontsize=17,color='navy')
ax[1].set_title('Last 11 sources')
plt.show()

#### Here we can see, clearly, that 1BY6WEcLGh8j5v7 & bvBOhCH3iADSZry are underperforming compared to other inverters,
# maybe these inverters require maintenance or require to be replaced. But before going into deep with underperforming inverters,
# let's look at which are the common problems for the entire plant,so let's see DC power generation during day hours for all 34 days.

temp1_gen=pg1.copy()

temp1_gen['time']=temp1_gen['DATE_TIME'].dt.time
temp1_gen['day']=temp1_gen['DATE_TIME'].dt.date

temp1_sens=ws1.copy()

temp1_sens['time']=temp1_sens['DATE_TIME'].dt.time
temp1_sens['day']=temp1_sens['DATE_TIME'].dt.date

# just for columns
cols=temp1_gen.groupby(['time','day'])['DC_POWER'].mean().unstack()



## DC POWER and DAILY YIELD in PLANT_1


ax =temp1_gen.groupby(['time','day'])['DC_POWER'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))
temp1_gen.groupby(['time','day'])['DAILY_YIELD'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,20),style='-.',ax=ax)

i=0
for a in range(len(ax)):
    for b in range(len(ax[a])):
        ax[a,b].set_title(cols.columns[i],size=15)
        ax[a,b].legend(['DC_POWER','DAILY_YIELD'])
        i=i+1

plt.tight_layout()
plt.show()

#### It seems that in some days there have been several problems with the plant,
# i.e. between 2020-05-19 and 2020-05-21 we can see a period which has null values that are common in daily yield and dc power generation.
# This may be due to a technical problem of the plant. Let's give a further look to ambient and module temperature:

## Module temperature and Ambient Temperature on PLANT_1

ax= temp1_sens.groupby(['time','day'])['MODULE_TEMPERATURE'].mean().unstack().plot(subplots=True,layout=(17,2),figsize=(20,30))
temp1_sens.groupby(['time','day'])['AMBIENT_TEMPERATURE'].mean().unstack().plot(subplots=True,layout=(17,2),figsize=(20,40),style='-.',ax=ax)

i=0
for a in range(len(ax)):
    for b in range(len(ax[a])):
        ax[a,b].axhline(50)
        ax[a,b].set_title(cols.columns[i],size=15)
        ax[a,b].legend(['Module Temperature','Ambient Temperature'])
        i=i+1

plt.tight_layout()
plt.show()

Well, it seems that in that period the plant doesn't work at all. So we must suppose that there was a technical problem in the plant.
Notice that a decrease in ambient temperature,just for a couple of degrees, influence quite a lot module temperature.

**P.S. I've drawn a line to see when module temperature goes over 50 degrees, this because a module overload may be the cause of a non-performing inverter.**


## Inverter bvBOhCH3iADSZry in action

#### As we can see between the 7th and 14th of June the dc power produced by the inverter goes quickly to 0 exactly during maximum sunlight hours,
# between 11 am and 16 pm. This can only be due to a fault in the inverter,
# so maybe these inverters requires to be fixed or replaced.

worst_source=gen_1[gen_1['SOURCE_KEY']=='bvBOhCH3iADSZry']
worst_source['time']=worst_source['DATE_TIME'].dt.time
worst_source['day']=worst_source['DATE_TIME'].dt.date

ax=worst_source.groupby(['time','day'])['DC_POWER'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))
worst_source.groupby(['time','day'])['DAILY_YIELD'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30),ax=ax,style='-.')

i=0
for a in range(len(ax)):
    for b in range(len(ax[a])):
        ax[a,b].set_title(cols.columns[i],size=15)
        ax[a,b].legend(['DC_POWER','DAILY_YIELD'])
        i=i+1

plt.tight_layout()
plt.show()

# Task 2: Forecast
## Can we predict the power generation for next couple of days?

#We're going to predict exactly the next two days of yield generated by plant_1, please note that we can have similar results predicting dc power generation rather then daily yield generated, but I think that for this purpose daily yield generated power is a good evidence of power prediction.

#We will tune, with auto_arima function, a SEASONAL ARIMA(p,d,q) + (P,D,Q,m) model,on the last 4 days(384 observations) to see if our model can capture the last generation trend.


from pandas.tseries.offsets import DateOffset
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller

pred_gen=pg1.copy()
pred_gen=pred_gen.groupby('DATE_TIME').sum()
pred_gen=pred_gen['DAILY_YIELD'][-288:].reset_index()
pred_gen.set_index('DATE_TIME',inplace=True)
pred_gen.head()


def train_test_split(df,panel_id_col,split_pct):

    # split into train test sets
    assert split_pct <= 1, 'Check percentage again'
    train_indices = []
    test_indices = []
    for panel_id in df[panel_id_col].unique():
        panel_id_index = df.loc[df[panel_id_col] == panel_id].index
        train = int(split_pct * len(panel_id_index))
        train_indices.append(panel_id_index[0:train])
        test_indices.append(panel_id_index[train:len(panel_id_index)])
    train_indices = [item for sublist in train_indices for item in sublist]
    test_indices = [item for sublist in test_indices for item in sublist]

    train_data = df.loc[train_indices].reset_index(drop=True)
    test_data = df.loc[test_indices].reset_index(drop=True)
    return train_data, test_data

#train-test split
plant1 = train_test_split(pg1,'SOURCE_KEY', 1)
plant1_train = plant1[0]
plant1_test = plant1[1]

## Step 3: Tune with the auto_arima function:
arima_model = auto_arima(plant1_train,
                         start_p=0,d=1,start_q=0,
                         max_p=4,max_d=4,max_q=4,
                         start_P=0,D=1,start_Q=0,
                         max_P=1,max_D=1,max_Q=1,m=96,
                         seasonal=True,
                         error_action='warn',trace=True,
                         supress_warning=True,stepwise=True,
                         random_state=20,n_fits=1)

## Step 4: Use the trained model which was built earlier to forecast daily yields

future_dates = [plant1_test.index[-1] + DateOffset(minutes=x) for x in range(0,2910,15) ]

prediction=pd.DataFrame(arima_model.predict(n_periods=96),index=plant1_test.index)
prediction.columns=['predicted_yield']

fig,ax= plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(17,5))
ax[0].plot(plant1_train,label='Train',color='navy')
ax[0].plot(plant1_test,label='Test',color='darkorange')
ax[0].plot(prediction,label='Prediction',color='green')
ax[0].legend()
ax[0].set_title('Forecast on test set',size=17)
ax[0].set_ylabel('kW',color='navy',fontsize=17)


f_prediction=pd.DataFrame(arima_model.predict(n_periods=194),index=future_dates)
f_prediction.columns=['predicted_yield']
ax[1].plot(pred_gen,label='Original data',color='navy')
ax[1].plot(f_prediction,label='18th & 19th June',color='green')
ax[1].legend()
ax[1].set_title('Next days forecast',size=17)
plt.show()


## Model summary:
arima_model.summary()



