""" Run RTKLib automatically for differential GNSS processing
http://www.rtklib.com/

input:  - GNSS options file (.conf)
        - GNSS rover file (rinex)
        - GNSS base file (rinex)
        - GNSS navigation ephemerides file (.nav); https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/broadcast_ephemeris_data.html#GPSdaily
        - GNSS precise ephemerides file (.eph/.sp3); https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/orbit_products.html
output: - position (.pos) file; (UTC, E, N, U)
created: LS

"""

# IMPORT modules
import subprocess
import os
import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# CHOOSE: DEFINE year, files (base, rover, navigation orbits, precise orbits), time interval
yy = str(21)
rover = 'NMER'      # 'NMER' or '3393' (old Emlid: 'ReachM2_sladina-raw_')
rover_name = 'NMER' # 'NMER' or 'NMER_original' or 'NMLR'
receiver = 'NMLR'
base = '3387'
nav = '3387'
sp3 = 'COD'
ti_int = '900'
resolution = '15min'
options_Leica = 'rtkpost_options_Ladina_Leica_statisch_multisystemfrequency_neumayer'
options_Emlid = 'rtkpost_options_Ladina_Emlid_statisch_multisystemfrequency_neumayer_900_15'

# Q: example run RTKLib:
# process1 = subprocess.Popen('cd data && rnx2rtkp -k rtkpost_options_Ladina.conf -ti 3600 -o sol/out_doy4.pos '
#                            'WJU10040.17O WJLR0040.17O alrt0040.17n COD17004.eph',
#                            shell= True,
#                            stdout=subprocess.PIPE,
#                            stderr=subprocess.PIPE)
#
# stdout1, stderr1 = process1.communicate()
# print(stdout1)
# print(stderr1)


# Q: For Emlid and Leica Rover
for file in glob.iglob('data_neumayer/' + rover + '*.' + yy + 'O', recursive=True):
    ''' get doy from rover file names with name structure:
        Leica Rover: '33933650.21o' [rover + doy + '0.' + yy + 'o']
        Emlid Rover (pre-processed): 'NMER3650.21o' [rover + doy + '0.' + yy + 'o']
        Emlid Rover (original): 'ReachM2_sladina-raw_202112041058.21O' [rover + datetime + '.' + yy + 'O']
        '''
    rover_file = os.path.basename(file)
    if rover_name == 'NMER_original':  # Emlid original format (output from receiver, non-daily files)
        doy = datetime.datetime.strptime(rover_file.split('.')[0].split('_')[2], "%Y%m%d%H%M").strftime('%j')
        options = options_Emlid
    if rover_name == 'NMER':       # Emlid pre-processed format (daily files)
        doy = rover_file.split('.')[0][-4:-1]
        options = options_Emlid
    if rover_name == 'NMLR':
        doy = rover_file.split('.')[0][-4:-1]
        options = options_Leica
    print('\nRover file: ' + rover_file, '\ndoy: ', doy)

    # define input and output filenames (for some reason it's not working when input files are stored in subfolders!)
    base_file = base + doy + '0.' + yy + 'O'
    broadcast_orbit_gps = nav + doy + '0.' + yy + 'n'
    broadcast_orbit_glonass = nav + doy + '0.' + yy + 'g'
    broadcast_orbit_galileo = nav + doy + '0.' + yy + 'l'
    precise_orbit = sp3 + yy + doy + '.sp3'
    output_file = 'sol/' + rover_name + '/20' + yy + '_' + rover_name + doy + '.pos'

    # run RTKLib automatically (instead of RTKPost Gui manually)
    process = subprocess.Popen('cd data_neumayer && rnx2rtkp '
                               '-k ' + options + '.conf '
                               '-ti ' + ti_int + ' '
                               '-o ' + output_file + ' '
                               + rover_file + ' ' + base_file + ' ' + broadcast_orbit_gps + ' ' + broadcast_orbit_glonass + ' ' + broadcast_orbit_galileo,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()
    # print(stdout) # print processing output
    print(stderr)  # print processing errors

    # remove .stat files
    if os.path.exists('data_neumayer/' + output_file + '.stat'):
        os.remove('data_neumayer/' + output_file + '.stat')

print('\n\nfinished with all files :-)')


''' import RTKLib solution .txt files '''
# create empty dataframe for all .ENU files
df_enu = pd.DataFrame()

# read all .ENU files in folder, parse date and time columns to datetimeindex and add them to the dataframe
for file in glob.iglob('data_neumayer/sol/' + receiver + '/*.pos', recursive=True):
    print(file)
    enu = pd.read_csv(file, header=24, delimiter=' ', skipinitialspace=True, index_col=['date_time'], na_values=["NaN"],
                      usecols=[0, 1, 4, 5, 6, 9], names=['date', 'time', 'U', 'amb_state', 'nr_sat', 'std_u'],
                      parse_dates=[['date', 'time']])
    df_enu = pd.concat([df_enu, enu], axis=0)

# store dataframe as binary pickle format
df_enu.to_pickle('data_neumayer/sol/' + receiver + '_' + resolution + '.pkl')


''' Read binary stored ENU data '''
# read all data from .pkl and combine, if necessary multiple parts
# df_enu = pd.read_pickle('data_neumayer/sol/' + receiver + '_' + resolution + '.pkl')

# select data where ambiguities are fixed (amb_state==1)
fil_df = pd.DataFrame(df_enu[(df_enu.amb_state == 1)])
fil_df.index = pd.DatetimeIndex(fil_df.index)
fil = ((fil_df.U - fil_df.U[1]).dropna()) * 1000 + 113  # adapt to reference SWE values in mm (median of last week without snow)

# remove outliers
upper_limit = fil.median() + 3 * fil.std()
lower_limit = fil.median() - 3 * fil.std()
fil_clean = fil[(fil > lower_limit) & (fil < upper_limit)]

# calculate median (per day and 10min) and std (per day)
m = fil_clean.resample('D').median()
s = fil_clean.resample('D').std()
m_15min = fil_clean.rolling('D').median()       # .resample('15min').median()
s_15min = fil_clean.rolling('D').std()

''' Read reference .csv data '''
# read SWE observations (30min resolution)
wfj = pd.read_csv('LLH/WFJ.csv', header=0, delimiter=';', index_col=0, na_values=["NaN"], names=['scale', 'pillow'])
wfj.index = pd.DatetimeIndex(wfj.index)
wfj = wfj.rolling('D').median()
wfj2 = wfj
wfj2.index = wfj2.index - pd.Timedelta(days=0.5)

# calculate median (per day and 10min) and relative bias (per day)
scale_res = scale.resample('D').median()
scale_err = scale_res/10     # calculate 10% relative bias
scale_10min = scale.rolling('D').median()   # calculate median per 10min (filtered over one day)

# combine daily manual and resampled scale observations in one dataframe
ref = pd.concat([scale_res, manual], axis=1)

# combine reference and GNSS data
all_daily = pd.concat([ref, m[:-1]], axis=1)
all_10min = pd.concat([scale_10min, m_15min], axis=1)

''' Plot results (SWE, ΔSWE, scatter)'''
m_ref = (ref['pillow'] + ref['scale'])/2 * 1000
ref['rtklib'] = ref['rtklib'] *1000
ref['pillow'] = ref['pillow']*1000
ref['scale'] = ref['scale']*1000

# plot SWE
plt.figure()
#ref['Manual'].plot(color='k', linestyle=' ', marker='+', markersize=8, markeredgewidth=2)
#plt.errorbar(ref['Manual'].index, ref['Manual'], yerr=ref['Manual']/10, color='k', capsize=4, alpha=0.5)
wfj2['pillow'].plot(linestyle='-', color='darkblue', fontsize=12, figsize=(6, 5.5), ylim=(-200, 1000))
wfj2['scale'].plot(linestyle='--', color='steelblue')
ref['rtklib'].plot(linestyle='-.', color='salmon')
m_15min.plot(color='crimson', linestyle='-').grid()
plt.fill_between(m_ref.index, m_ref - m_ref/10, m_ref + m_ref/10, color="darkblue", alpha=0.1)
plt.fill_between(s_15min.index, m_15min - s_15min, m_15min + s_15min, color="crimson", alpha=0.1)
plt.xlabel(None)
plt.ylabel('SWE (mm w.e.)', fontsize=14)
plt.legend(['Snow pillow', 'Snow scale', 'High-end GNSS', 'Low-cost GNSS'], fontsize=12, loc='upper left')
plt.xlim(datetime.datetime.strptime('2016-11-01', "%Y-%m-%d"), datetime.datetime.strptime('2017-07-01', "%Y-%m-%d"))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.show()
plt.savefig('LLH/SWE_WFJ.png', bbox_inches='tight')
plt.savefig('LLH/SWE_WFJ.pdf', bbox_inches='tight')

# plot SWE difference
plt.close()
plt.figure()
(ref['scale']-ref['rtklib']).plot(color='steelblue', linestyle='--', fontsize=14, figsize=(6, 5.5), ylim=(-200, 200)).grid()
(ref['pillow']-ref['rtklib']).plot(color='darkblue', linestyle='-').grid()
plt.xlabel(None)
plt.ylabel('ΔSWE (mm w.e.)', fontsize=14)
plt.legend(['Snow scale', 'Snow pillow'], fontsize=14, loc='upper left')
plt.xlim(datetime.datetime.strptime('2016-11-01', "%Y-%m-%d"), datetime.datetime.strptime('2017-07-01', "%Y-%m-%d"))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.show()
plt.savefig('LLH/diff_SWE_WFJ_highend.png', bbox_inches='tight')
plt.savefig('LLH/diff_SWE_WFJ_highend.pdf', bbox_inches='tight')

# fit linear regression curve manual vs. GNSS (daily)
all_daily_nonan = all_daily.dropna()
fit = np.polyfit(all_daily_nonan['Manual'], all_daily_nonan['U'], 1)
predict = np.poly1d(fit)
print('Linear fit: \nm = ', round(fit[0], 2), '\nb = ', int(fit[1]))     # n=12, m=1.02, b=-8 mm w.e.

# calculate cross correation manual vs. GNSS (daily)
corr = all_daily['Manual'].corr(all_daily['U'])
print('Pearsons correlation: %.2f' % corr)

# plot scatter plot (GNSS vs. manual, daily)
plt.close()
plt.figure()
ax = all_daily.plot.scatter(x='Manual', y='U', figsize=(5, 4.5))
plt.plot(range(10, 360), predict(range(10, 360)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(0, 500)
ax.set_xlim(0, 500)
ax.set_xlabel('Manual SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
# plt.savefig('ENU/scatter_SWE_manual.png', bbox_inches='tight')
# plt.savefig('ENU/scatter_SWE_manual.pdf', bbox_inches='tight')


# fit linear regression curve scale vs. GNSS (10min)
all_10min_nonan = all_10min.dropna()
fit = np.polyfit(all_10min_nonan['Scale'], all_10min_nonan['U'], 1)
predict = np.poly1d(fit)
print('Linear fit: \nm = ', round(fit[0], 2), '\nb = ', int(fit[1]))     # n=11526, m=0.99, b=-36 mm w.e.

# calculate cross correation scale vs. GNSS (10min)
corr = all_10min['Scale'].corr(all_10min['U'])
print('Pearsons correlation: %.2f' % corr)


# plot scatter plot (GNSS vs. scale, 10min data)
plt.close()
plt.figure()
ax = all_10min.plot.scatter(x='Scale', y='U', figsize=(5, 4.5))
plt.plot(range(10, 390), predict(range(10, 390)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(0, 500)
ax.set_xlim(0, 500)
ax.set_xlabel('Snow scale SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
# plt.savefig('ENU/scatter_SWE_scale_10min.png', bbox_inches='tight')
# plt.savefig('ENU/scatter_SWE_scale_10min.pdf', bbox_inches='tight')


# fit linear regression curve scale vs. GNSS (10min)
libref = ref.dropna()
fit = np.polyfit(libref['pillow'].dropna(), libref['rtklib'].dropna(), 1)
predict = np.poly1d(fit)
print('Linear fit: \nm = ', round(fit[0], 2), '\nb = ', int(fit[1]))     # n=11526, m=0.99, b=-36 mm w.e.

# calculate cross correation scale vs. GNSS (10min)
corr = ref['pillow'].corr(ref['rtklib'])
print('Pearsons correlation: %.2f' % corr)


# plot scatter plot (GNSS vs. scale, 10min data)
plt.close()
plt.figure()
ax = ref.plot.scatter(x='pillow', y='rtklib', figsize=(5, 4.5))
plt.plot(range(0, 800), predict(range(0, 800)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('High-end GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(-1, 1000)
ax.set_xlim(-1, 1000)
ax.set_xlabel('Snow pillow SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
#plt.savefig('LLH/scatter_SWE_WFJ_pillow_highend.png', bbox_inches='tight')
#plt.savefig('LLH/scatter_SWE_WFJ_pillow_highend.pdf', bbox_inches='tight')
