import pandas as pd
import sys
import warnings

warnings.filterwarnings('ignore')

input_srvy = sys.argv[1]
output_srvy = sys.argv[2]

print('Reading 2023 raw survey....')
srvy = pd.read_excel(input_srvy)
print('2023 raw survey size: ',srvy.shape)

# identify columns that need to be dropped
cols_home = ['HOME_ADDRESS [LAT]','HOME_ADDRESS [LONG]','ORIGIN_ADDRESS [LAT]','ORIGIN_ADDRESS [LONG]','DESTIN_ADDRESS [LAT]','DESTIN_ADDRESS [LONG]']
cols_onoff = ['STOP_ON [LAT]','STOP_ON [LONG]','STOP_OFF [LAT]','STOP_OFF [LONG]']
cols_interim_onoff = [
    'PREV_TRAN_1_ON_BUS [LAT]',
    'PREV_TRAN_1_ON_BUS [LONG]',
    'PREV_TRAN_1_OFF_BUS [LAT]',
    'PREV_TRAN_1_OFF_BUS [LONG]',
    'PREV_TRAN_2_ON_BUS [LAT]',
    'PREV_TRAN_2_ON_BUS [LONG]',
    'PREV_TRAN_2_OFF_BUS [LAT]',
    'PREV_TRAN_2_OFF_BUS [LONG]',
    'PREV_TRAN_3_ON_BUS [LAT]',
    'PREV_TRAN_3_ON_BUS [LONG]',
    'PREV_TRAN_3_OFF_BUS [LAT]',
    'PREV_TRAN_3_OFF_BUS [LONG]',
    'PREV_TRAN_4_ON_BUS [LAT]',
    'PREV_TRAN_4_ON_BUS [LONG]',
    'PREV_TRAN_4_OFF_BUS [LAT]',
    'PREV_TRAN_4_OFF_BUS [LONG]',
    'NEXT_TRAN_1_ON_BUS [LAT]',
    'NEXT_TRAN_1_ON_BUS [LONG]',
    'NEXT_TRAN_1_OFF_BUS [LAT]',
    'NEXT_TRAN_1_OFF_BUS [LONG]',
    'NEXT_TRAN_2_ON_BUS [LAT]',
    'NEXT_TRAN_2_ON_BUS [LONG]',
    'NEXT_TRAN_2_OFF_BUS [LAT]',
    'NEXT_TRAN_2_OFF_BUS [LONG]',
    'NEXT_TRAN_3_ON_BUS [LAT]',
    'NEXT_TRAN_3_ON_BUS [LONG]',
    'NEXT_TRAN_3_OFF_BUS [LAT]',
    'NEXT_TRAN_3_OFF_BUS [LONG]',
    'NEXT_TRAN_4_ON_BUS [LAT]',
    'NEXT_TRAN_4_ON_BUS [LONG]',
    'NEXT_TRAN_4_OFF_BUS [LAT]',
    'NEXT_TRAN_4_OFF_BUS [LONG]'
]

cols_to_drop = cols_home + cols_onoff + cols_interim_onoff

srvy = srvy.drop(columns=cols_to_drop)

print('2023 no PII survey size: ',srvy.shape)
print('Exporting 2023 no PII survey....')

srvy.to_excel(output_srvy,index=False)
print(f'Done. 2023 no PII version is saved here: {output_srvy}')
