""" Run RTKLib automatically for differential GNSS processing
https://www.rtklib.com/

input:  - GNSS options file (.conf)
        - GNSS rover file (rinex)
        - GNSS base file (rinex)
        - GNSS navigation ephemerides file (.nav); https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/broadcast_ephemeris_data.html#GPSdaily
        - GNSS precise ephemerides file (.eph/.sp3); https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/orbit_products.html
output: - position (.pos) file; (UTC, X, Y, Z)
created: LS

"""

# IMPORT modules
import subprocess

# DEFINE year, files(base, rover, navigation orbits, precise orbits), time interval
yy: str = str(17)
base = 'WJLR'
rover = 'WJU1'
nav = 'alrt'
sp3 = 'COD'
ti_int = '3600'
options = 'rtkpost_options_Ladina'


# example: 
# process1 = subprocess.Popen('cd data && rnx2rtkp -k rtkpost_options_Ladina.conf -ti 3600 -o solutions/out_doy4.pos '
#                            'WJU10040.17O WJLR0040.17O alrt0040.17n COD17004.eph',
#                            shell= True,
#                            stdout=subprocess.PIPE,
#                            stderr=subprocess.PIPE)
#
# stdout1, stderr1 = process1.communicate()
# print(stdout1)
# print(stderr1)


# iterator for 3-digit numbers (001 etc.)
doy_list = ["%.3d" % i for i in range(1, 8)]

# for each day of year, do:
for doy in doy_list:
    doy = str(doy)
    print('doy: ', doy, doy[-1])
    
    # run RTKLib automatically (instead of RTKPost Gui manually)
    process = subprocess.Popen('cd data && rnx2rtkp -k ' + options + '.conf -ts 20' + yy + '/01/0' + doy[-1] + ' 00:00:00 -te 20'
                               + yy + '/01/0' + doy[-1] + ' 23:59:59 -ti ' + ti_int + ' -o solutions/out' + doy + '.pos '
                               + rover + doy + '0.' + yy + 'O ' + base + doy + '0.' + yy + 'O alrt' + doy + '0.' + yy + 'n ' + sp3 + yy + doy + '.eph',
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)
