from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from sklearn.decomposition import FastICA
import requests
from tqdm import tqdm
from more_itertools import locate

def import_data():
    sb2_fits = fits.open("/Users/blaine/Documents/astrodata/apogee_sb2s-v1_0.fits", memmap=True)
    a = sb2_fits[1].data
    all_visit = fits.open("/Users/blaine/Documents/astrodata/allVisit-dr17-synspec_rev1.fits", memmap=True)
    b = all_visit[1].data
    all_star = fits.open("/Users/blaine/Documents/astrodata/allStar-dr17-synspec_rev1.fits", memmap=True)
    c = all_star[1].data

    return a,b,c

def filter_sb2(sb2):
    ret = []
    for i in range(0, sb2.shape[0]):
        arr = sb2[i][10]
        if arr[0] >= 3 and arr[1] >= 3 and arr[2] == 0 and arr[3] == 0:
            ret.append(i)
    return ret

def filter_allVisit(visit_id, visit, id_lst):
    ret = []
    for dat in tqdm(id_lst):
        dat_mjd = dat[0]
        dat_id = dat[1]
        temp = list(locate(visit_id[offset:-1], lambda x:x[1] == dat_mjd and x[0] == dat_id))
        if len(temp) > 0:
            idx = temp[0]
            ret.append([visit[idx][3], idx])
        else:
            continue
    return ret

def dict_display(dict,N):
    for i in range(0,N):
        print(list(dict.items())[i])
        print("")


pip install numpy==1.21.5


sb2_raw, visit_raw, star_raw = import_data()
#Filter sb2 data to have both ratings above 3
sb2_idx = filter_sb2(sb2_raw)


#Condense SB2 Data into a list of all filtered elements
sb2_data = [sb2_raw[idx] for idx in sb2_idx]


#Create a Dictionary of all elements of sb2 logged by appogee ID
sb2_mjd = [[int(np.floor(element[1])), element] for element in sb2_data]


#Convert fits format to list of data
visit_id_list = [[visit[0], visit[7]] for visit in visit_raw]


sb2_info_list = [[b2[0],b2[1][0]] for b2 in sb2_mjd] 


visit_raw[656][3]


filtered_allVisit = filter_allVisit(visit_id_list, visit_raw, sb2_info_list[0:4])


#Get the index of all_visit data that matches sb2_appid_lst
filtered_allVisit = filter_allVisit(visit_id_list, visit_raw, sb2_info_list[0:100])


filtered_allVisit


len(sb2_mjd)


allvisit_mjd_list = [x[7] for x in filtered_allVisit]
unique_all_visit = list(dict.fromkeys(allvisit_mjd_list))


a_visit_file_name = []
un = []
cnt = 0
for mjd in tqdm(sb2_mjd):
    try:
        idx = allvisit_mjd_list.index(mjd[0])
        a_visit_file_name.append([filtered_allVisit[idx][0],mjd[0],filtered_allVisit[idx][3]])
    except ValueError:
        cnt += 1
        continue


sb2_mjd[7]


sb2_mjd[39]


a_visit_file_name


temp_1


start = 0
for i in tqdm(range(0,len(temp_1))):
    try:
        mjd_0 = all_sb2_mjd[i]
        idx = all_visit_mjd.index(mjd_0)
        extract_data_0 = binary[idx]
        extract_data_1 = filtered_sb2_data[i]
        
        filename = str(extract_data_0[3])
        mjd = int(extract_data_0[7])
        apogeeID = extract_data_1[0]
        visitno = int(extract_data_0[4])
        try:
            testing_data = fits.open("../../../data/" + extract_data_0[3], memmap=True)
        except FileNotFoundError:
            continue
        fluxa = [float(a) for a in testing_data[1].data[0]]
        wavelengtha = [float(a) for a in testing_data[4].data[0]]
        fluxb = [float(a) for a in testing_data[1].data[1]]
        wavelengthb = [float(a) for a in testing_data[4].data[1]]
        fluxc = [float(a) for a in testing_data[1].data[2]]
        wavelengthc = [float(a) for a in testing_data[4].data[2]]
        ampa = float(extract_data_1[4][0])
        vhelioa = float(extract_data_1[5][0])
        fwhma = float(extract_data_1[6][0])
        ampb = float(extract_data_1[4][1])
        vheliob = float(extract_data_1[5][1])
        fwhmb = float(extract_data_1[6][1])
        SNR = float(extract_data_0["SNR"])
        if np.isnan(ampa) or np.isnan(SNR) or np.isnan(vhelioa) or np.isnan(ampa) or np.isnan(fwhma) or np.isnan(ampb) or np.isnan(vheliob) or np.isnan(fwhmb):
            continue
        attr = ["filename", "apogeeID", "visitno", "fluxa", "wavelengtha", "fluxb", "wavelengthb", "fluxc", "wavelengthc",
                     "ampa", "vhelioa", "fwhma", "ampb", "vheliob", "fwhmb", "SNR", "mjd"]
        vals = [filename, apogeeID, visitno, fluxa, wavelengtha, fluxb, wavelengthb, fluxc, wavelengthc,
                         ampa, vhelioa, fwhma, ampb, vheliob, fwhmb, SNR, mjd]
        to_json = {a:b for a,b in zip(attr,vals)}
        requests.post('http://127.0.0.1:5000/test', json=to_json)
    except ValueError:
        print("bruh")
        pass



