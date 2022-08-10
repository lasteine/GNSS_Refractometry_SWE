""" Run RTKLib automatically for differential GNSS post processing and SWE estimation
http://www.rtklib.com/

Reference: Steiner et al., (Near) Real-Time Snow Water Equivalent Observation Using GNSS Refractometry and RTKLIB, submitted to Sensors, 2022.

input:  - GNSS config file (.conf)
        - GNSS rover file (rinex)
        - GNSS base file (rinex)
        - GNSS navigation ephemerides file (.nav); https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/broadcast_ephemeris_data.html#GPSdaily
        - GNSS precise ephemerides file (.eph/.sp3); https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/orbit_products.html

output: - position (.pos) file; (UTC, E, N, U)
        - plots (SWE timeseries, DeltaSWE timeseries, scatter plots)

created: L. Steiner (Orchid ID: 0000-0002-4958-0849)
date:    8.8.2022
"""

# IMPORT modules
import subprocess
import os
import datetime as dt
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gnsscal  # https://github.com/purpleskyfall/gnsscal, Copyright (c) 2017, Jon Jiang

# CHOOSE: year, files (base, rover, navigation orbits, precise orbits, config file), time interval
yy = str(17)                    # year
base = 'WJLR'                   # name prefix of base file
rover = 'WJL0'                  # name prefix of rover file
nav = 'alrt'                    # name prefix of broadcast nav file
sp3 = 'COD'                     # name prefix of precise orbit file
ti_int = '900'                  # time interval for post processing in seconds
resolution = '15min'            # processing resolution in minutes
options = 'rtkpost_options'     # rtklib post processing configuration file
# define start and end day of year
start_doy = 0
end_doy = 366


""" 1. run RTKLib automatically (instead of RTKPost Gui manually) """
# Q: run rtklib for all rover files in directory
for file in glob.iglob('data/' + rover + '*.' + yy + 'O', recursive=True):
    ''' get doy from rover file names with name structure:
        Leica Rover: 'WJL03650.21o' [rover + doy + '0.' + yy + 'o']
        '''

    # get rover file and doy from rover files in directory
    rover_file = os.path.basename(file)
    doy = rover_file.split('.')[0][4:7]
    print('\nRover file: ' + rover_file, '\ndoy: ', doy)

    if int(doy) >= start_doy & int(doy) <= end_doy:

        # convert doy to gpsweek and day of week
        (gpsweek, dow) = gnsscal.yrdoy2gpswd(int('20' + yy), doy)

        # Q: define input and output filenames
        base_file = base + doy + '0.' + yy + 'O'
        broadcast_orbit_gps = nav + doy + '0.' + yy + 'n'
        broadcast_orbit_glonass = nav + doy + '0.' + yy + 'g'
        broadcast_orbit_galileo = nav + doy + '0.' + yy + 'l'
        precise_orbit = sp3 + str(gpsweek) + str(dow) + '.EPH_M'
        output_file = 'sol/' + rover + '/' + resolution + '/' + rover + doy + '.pos'

        # Q: change directory & run RTKLib post processing command
        process = subprocess.Popen('cd data && rnx2rtkp '
                                   '-k ' + options + '.conf '
                                   '-ti ' + ti_int + ' '
                                   '-o ' + output_file + ' '
                                   + rover_file + ' ' + base_file + ' ' + broadcast_orbit_gps + ' ' + broadcast_orbit_glonass + ' ' + broadcast_orbit_galileo + ' ' + precise_orbit,
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()
        # print(stdout) # print processing output
        print(stderr)  # print processing errors

print('\n\nfinished post processing all files in directory :-)')


""" 2. Get rtklib ENU solution files"""
# create empty dataframe for all .ENU solution files
df_enu = pd.DataFrame()

# Q read all .ENU files in folder, parse date and time columns to datetimeindex and add them to the dataframe
for file in glob.iglob('data/sol/' + rover + '/' + resolution + '/*.pos', recursive=True):
    print(file)
    enu = pd.read_csv(file, header=24, delimiter=' ', skipinitialspace=True, index_col=['date_time'], na_values=["NaN"],
                      usecols=[0, 1, 4, 5, 6, 9], names=['date', 'time', 'U', 'amb_state', 'nr_sat', 'std_u'],
                      parse_dates=[['date', 'time']])
    df_enu = pd.concat([df_enu, enu], axis=0)

# store dataframe as binary pickle format
# df_enu.to_pickle('data/sol/' + rover + '_' + resolution + '.pkl')


''' 3. Filter and clean ENU solution data '''
# read all data from .pkl and combine, if necessary multiple parts
# df_enu = pd.read_pickle('data/sol/' + rover + '_' + resolution + '.pkl')

# select only data where ambiguities are fixed (amb_state==1) and sort datetime index
fil_df = pd.DataFrame(df_enu[(df_enu.amb_state == 1)])
fil_df.index = pd.DatetimeIndex(fil_df.index)
fil_df = fil_df.sort_index()

# adapt up values to reference SWE values in mm (median of last week without snow)
fil = (fil_df.U - fil_df.U[-900:].median()) * 1000

# remove outliers based on a 3*sigma threshold
upper_limit = fil.median() + 3 * fil.std()
lower_limit = fil.median() - 3 * fil.std()
fil_clean = fil[(fil > lower_limit) & (fil < upper_limit)]

# resample data, calculate median and standard deviation (noise) per day to fit manual reference data
m = fil_clean.resample('D').median()
s = fil_clean.resample('D').std()

# filter data with a rolling median and resample resolution to fit reference data (30min)
m_30min = fil_clean.rolling('D').median().resample('30min').mean()


''' 3. Read reference sensors .csv data '''
# read snow depth observations (hourly resolutions)
sh = pd.read_csv('data/ref/wfj_sh.csv', header=0, delimiter=';', index_col=0, na_values=["NaN"], parse_dates=[0])

# read manual SWE observations (monthly resolution)
manual = pd.read_csv('data/ref/manual.csv', header=None, skipinitialspace=True, delimiter=';', index_col=0, skiprows=0, na_values=["NaN"], parse_dates=[0], dayfirst=True, usecols=[0, 2], names=['date', 'manual'])

# read automated SWE observations (30min resolution)
wfj = pd.read_csv('data/ref/WFJ.csv', header=0, delimiter=';', index_col=0, na_values=["NaN"], names=['scale', 'pillow'])
wfj.index = pd.DatetimeIndex(wfj.index)
wfj.index = wfj.index.tz_convert(None)          # convert to timezone unaware index
scale_30min = wfj.scale.rolling('D').median()   # calculate median (per 30min)
scale_30min.index = scale_30min.index - pd.Timedelta(days=0.5)  # correct index after rolling median filter
scale_res = scale_30min.resample('D').median()  # calculate median (per day)
scale_err = scale_res/10                        # calculate 10% relative bias


""" 4. Calculate differences, linear regressions, RMSE & MRB between GNSS and reference data """
# Q: calculate differences between GNSS and reference data
dmanual = (manual.manual - m).dropna()      # daily
dscale_daily = (scale_res - m).dropna()     # daily
dscale = (scale_30min - m_30min).dropna()   # 30min
diffs = pd.concat([dmanual, dscale], axis=1)
diffs.columns = ['Manual', 'Snow scale']

# Q: cross correlation and linear fit (daily & 30min)
# merge manual and gnss data (daily)
all_daily = pd.concat([manual.manual, m], axis=1)
all_daily_nonan = all_daily.dropna()
# merge scale and gnss data (30min)
all_30min = pd.concat([scale_30min, m_30min], axis=1)
all_30min_nonan = all_30min.dropna()

# cross correation manual vs. GNSS (daily)
corr_daily = all_daily.manual.corr(all_daily.U)
print('\nPearsons correlation (manual vs. GNSS, daily): %.2f' % corr_daily)
# calculate cross correation scale vs. GNSS (30min)
corr_30min = all_30min.scale.corr(all_30min.U)
print('Pearsons correlation (scale vs. GNSS, 30min): %.2f' % corr_30min)

# fit linear regression curve manual vs. GNSS (daily)
fit_daily = np.polyfit(all_daily_nonan.manual, all_daily_nonan.U, 1)
predict_daily = np.poly1d(fit_daily)
print('\nLinear fit (manual vs. GNSS, daily): \nm = ', round(fit_daily[0], 2), '\nb = ', int(fit_daily[1]))
# fit linear regression curve scale vs. GNSS (30min)
fit_30min = np.polyfit(all_30min_nonan.scale, all_30min_nonan.U, 1)
predict_30min = np.poly1d(fit_30min)
print('Linear fit (scale vs. GNSS, 30min): \nm = ', round(fit_30min[0], 2), '\nb = ', int(fit_30min[1]))     # n=12, m=1.02, b=-8 mm w.e.

# RMSE
rmse_manual = np.sqrt((np.sum(dmanual**2))/len(dmanual))
print('\nRMSE (manual vs. GNSS, daily): %.1f' % rmse_manual)
rmse_scale = np.sqrt((np.sum(dscale**2))/len(dscale))
print('RMSE (scale vs. GNSS, 30min): %.1f' % rmse_scale)

# MRB
mrb_manual = (dmanual/all_daily.manual).mean() * 100
print('\nMRB (manual vs. GNSS, daily): %.1f' % mrb_manual)
mrb_scale = (dscale/all_30min.scale).mean() * 100
print('MRB (scale vs. GNSS, 30min): %.1f' % mrb_scale)

# Number of samples
n_manual = len(dmanual)
print('\nNumber of samples (manual vs. GNSS, daily): %.0f' % n_manual)
n_scale = len(dscale)
print('Number of samples (scale vs. GNSS, 30min): %.0f' % n_scale)


''' 5. Plot results (SWE, ΔSWE, scatter) '''
# Q: plot SWE
plt.figure()
ax = scale_res.plot(linestyle='--', color='steelblue', fontsize=12, figsize=(6, 5.5), ylim=(-1, 1000))
ax2 = ax.twinx()
ax2.plot(sh.rolling('D').median()/10, color='darkgrey') # plot snow depth on right axis

manual.manual.plot(color='k', linestyle=' ', marker='+', markersize=8, markeredgewidth=2, ax=ax)
ax.errorbar(manual.index, manual.manual, yerr=manual.manual/10, linestyle=' ', color='k', capsize=4, alpha=0.5)
m.plot(color='crimson', ax=ax)
ax.fill_between(m.index, m - s, m + s, color="crimson", alpha=0.15)
ax.fill_between(scale_res.index, scale_res - scale_err, scale_res + scale_err, color="steelblue", alpha=0.1)

# set left axis limits, labels, params, legends
ax.set_xlim([dt.date(2016, 11, 1), dt.date(2017, 7, 1)])
ax.set_ylim(0, 1000) # 1/4 of scale of right y-axes, for sharing 0 and scale
ax.set_xlabel(None)
ax.set_ylabel('SWE (mm w.e.)', color='k', fontsize=14)
ax.grid(True)
ax.tick_params(axis="y", colors='k', labelsize=12)
ax.legend(['Snow scale', 'Manual', 'GNSS'], fontsize=14, loc='upper left')

# set right axis limits, labels, params, no legend
ax2.tick_params(colors='darkgrey', labelsize=12)
ax2.set_xlim([dt.date(2016, 11, 1), dt.date(2017, 7, 1)])
ax2.set_ylim(0,250)
ax2.set_ylabel('Snow depth (cm)', color='darkgrey', fontsize=14)
plt.show()
# plt.savefig('ENU/SWE_SH_WFJ_highend.png', bbox_inches='tight')
# plt.savefig('ENU/SWE_SH_WFJ_highend.pdf', bbox_inches='tight')


# Q. plot SWE difference
plt.close()
plt.figure()
dscale_daily.plot(color='steelblue', linestyle='--', fontsize=14, figsize=(6, 5.5), ylim=(-200, 200)).grid()
dmanual.plot(color='k', linestyle=' ', marker='+', markersize=8, markeredgewidth=2).grid()
plt.xlabel(None)
plt.ylabel('ΔSWE (mm w.e.)', fontsize=14)
plt.legend(['Snow scale', 'Manual'], fontsize=14, loc='upper left')
plt.xlim([dt.date(2016, 11, 1), dt.date(2017, 7, 1)])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
# plt.savefig('plots/diff_SWE_WFJ_highend.png', bbox_inches='tight')
# plt.savefig('plots/diff_SWE_WFJ_highend.pdf', bbox_inches='tight')


# Q: plot scatter plot (GNSS vs. manual, daily)
plt.close()
plt.figure()
ax = all_daily.plot.scatter(x='manual', y='U', figsize=(5, 4.5))
plt.plot(range(10, 750), predict_daily(range(10, 750)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(0, 1000)
ax.set_xlim(0, 1000)
ax.set_xlabel('Manual SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr_daily], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
# plt.savefig('plots/scatter_SWE_WFJ_manual.png', bbox_inches='tight')
# plt.savefig('plots/scatter_SWE_WFJ_manual.pdf', bbox_inches='tight')


# Q: plot scatter plot (GNSS vs. scale, 30min)
plt.close()
plt.figure()
ax = all_30min.plot.scatter(x='scale', y='U', figsize=(5, 4.5))
plt.plot(range(10, 850), predict_30min(range(10, 850)), c='k', linestyle='--', alpha=0.7)    # linear regression
ax.set_ylabel('GNSS SWE (mm w.e.)', fontsize=12)
ax.set_ylim(0, 1000)
ax.set_xlim(0, 1000)
ax.set_xlabel('Snow scale SWE (mm w.e.)', fontsize=12)
plt.legend(['r=%.2f' % corr_30min], fontsize=12, loc='upper left')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.show()
# plt.savefig('plots/scatter_SWE_WFJ_scale_30min.png', bbox_inches='tight')
# plt.savefig('plots/scatter_SWE_WFJ_scale_30min.pdf', bbox_inches='tight')

# Q: plot boxplot of differences
dscale.describe()
dmanual.describe()
diffs[['Manual', 'Snow scale']].plot.box(ylim=(-100, 200), figsize=(3, 4.5), fontsize=12)
plt.grid()
plt.ylabel('ΔSWE (mm w.e.)', fontsize=12)
plt.show()
# plt.savefig('plots/box_SWE_WFJ_diff.png', bbox_inches='tight')
# plt.savefig('plots/box_SWE_WFJ_diff.pdf', bbox_inches='tight')

# Q: plot histogram of differences
diffs[['Snow scale', 'Manual']].plot.hist(bins=25, xlim=(-200, 200), figsize=(3, 4.5), fontsize=12, alpha=0.8)
plt.grid()
plt.xlabel('ΔSWE (mm w.e.)', fontsize=12)
plt.legend(loc='upper left')
plt.show()
# plt.savefig('ENU/hist_SWE_diff.png', bbox_inches='tight')
# plt.savefig('ENU/hist_SWE_diff.pdf', bbox_inches='tight')



