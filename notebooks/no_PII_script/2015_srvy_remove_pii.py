import pandas as pd
import sys
import warnings

warnings.filterwarnings('ignore')

input_srvy = sys.argv[1]
output_srvy = sys.argv[2]

print('Reading 2015 raw survey....')
srvy = pd.read_excel(input_srvy)
print('2015 raw survey size: ',srvy.shape)

# identify columns that need to be dropped
cols_home = ['HOME_OR_HOTEL_ADDR_LAT','HOME_OR_HOTEL_ADDR_LON','ORIGIN_ADDRESS_LAT','ORIGIN_ADDRESS_LON','DESTIN_ADDRESS_LAT','DESTIN_ADDRESS_LON',\
             'HOME_OR_HOTEL_ADDRESS','ORIGIN_ADDRESS','DESTIN_ADDRESS']
cols_work = ['Q17A_WORK_ADDR_LAT','Q17A_WORK_ADDR_LON']
cols_onoff = ['BOARDING_LAT','BOARDING_LON','ALIGHTING_LAT','ALIGHTING_LON']
cols_pnr = ['VEHICLE_ACCESS_DROPOFF_LOCATION_LAT','VEHICLE_ACCESS_DROPOFF_LOCATION_LON','VEHICLE_EGRESS_DROPOFF_LOCATION_LAT','VEHICLE_EGRESS_DROPOFF_LOCATION_LON']

cols_to_drop = cols_home + cols_onoff + cols_work + cols_pnr

srvy = srvy.drop(columns=cols_to_drop)

print('2015 no PII survey size: ',srvy.shape)
print('Exporting 2015 no PII survey....')

srvy.to_excel(output_srvy,index=False)
print(f'Done. 2015 no PII version is saved here: {output_srvy}')
