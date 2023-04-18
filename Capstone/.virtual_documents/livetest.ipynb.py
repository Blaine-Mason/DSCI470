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


#Get the index of all_visit data that matches sb2_appid_lst
#filtered_allVisit = filter_allVisit(visit_id_list, visit_raw, sb2_info_list)


f = open("data_visits.txt", "r")
temp = f.readlines()
filtered_allVisit = []
for line in temp:
    el = line
    el = el.replace("[","")
    el = el.replace("]","")
    el = el.replace("'","")
    el = el.replace(" ","")
    final = el.split(",")
    final[1] = int(final[1])
    filtered_allVisit.append(final)


filenames = [e[0] for e in  filtered_allVisit]


filtered_allVisit


filenames


visit_raw[656]["SNR"]


allvisit_mjd_list = [visit_raw[x[1]][7] for x in filtered_allVisit]


len(allvisit_mjd_list)


unique_all_visit = list(dict.fromkeys(allvisit_mjd_list))


len(unique_all_visit)


a_visit_file_name = []
un = []
cnt = 0
for mjd in tqdm(sb2_mjd):
    try:
        idx = allvisit_mjd_list.index(mjd[0])
        a_visit_file_name.append([visit_raw[filtered_allVisit[idx][1]][0],mjd[0],filtered_allVisit[idx][0]])
    except ValueError:
        cnt += 1
        continue


t = [x[2] for x in a_visit_file_name]
t = list(dict.fromkeys(t))


f = open("visit_installation.txt", "a")
cnt = 0
for a in tqdm(t):
    one = a.split("-")[2]
    two = a.split("-")[3]
    f.write("https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/plates/" + one + "/" + two + "/" + a + "\n")
f.close()


temp_list = [str(a) for a in a_visit_file_name]
temp_list = list(dict.fromkeys(temp_list))


list_for_db = []
for g in temp_list:
    el = g
    el = el.replace("[","")
    el = el.replace("]","")
    el = el.replace("'","")
    el = el.replace(" ","")
    final = el.split(",")
    final[1] = int(final[1])
    list_for_db.append(final)


list_for_db


temp = 0
for lst in tqdm(list_for_db):
    try:
        filename = str(lst[2])
        mjd = int(lst[1])
        apogeeID = lst[0]
        try:
            testing_data = fits.open("/Users/bmason3/astrodata/data2/" + filename, memmap=True)
        except FileNotFoundError:
            continue
        fluxa = [float(a) for a in testing_data[1].data[0]]
        wavelengtha = [float(a) for a in testing_data[4].data[0]]
        fluxb = [float(a) for a in testing_data[1].data[1]]
        wavelengthb = [float(a) for a in testing_data[4].data[1]]
        fluxc = [float(a) for a in testing_data[1].data[2]]
        wavelengthc = [float(a) for a in testing_data[4].data[2]]
        
        tmplst = [mjd, apogeeID]
        tempidx = list(locate(sb2_info_list, lambda x:x == tmplst))
        allvisit_temp_idx = list(locate(filtered_allVisit, lambda x:x[0] == filename))
        allvisit_idx = filtered_allVisit[allvisit_temp_idx[0]][1]
        extract_data_0 = visit_raw[allvisit_idx]
        extract_data_1 = sb2_data[tempidx[0]]
        
        ampa = float(extract_data_1[4][0])
        vhelioa = float(extract_data_1[5][0])
        fwhma = float(extract_data_1[6][0])
        ampb = float(extract_data_1[4][1])
        vheliob = float(extract_data_1[5][1])
        fwhmb = float(extract_data_1[6][1])
        SNR = float(extract_data_0["SNR"])
        if np.isnan(ampa) or np.isnan(SNR) or np.isnan(vhelioa) or np.isnan(ampa) or np.isnan(fwhma) or np.isnan(ampb) or np.isnan(vheliob) or np.isnan(fwhmb):
            continue
        attr = ["temp", "filename", "apogeeID", "fluxa", "wavelengtha", "fluxb", "wavelengthb", "fluxc", "wavelengthc",
                     "ampa", "vhelioa", "fwhma", "ampb", "vheliob", "fwhmb", "SNR", "mjd"]
        vals = [temp, filename, apogeeID, fluxa, wavelengtha, fluxb, wavelengthb, fluxc, wavelengthc,
                         ampa, vhelioa, fwhma, ampb, vheliob, fwhmb, SNR, mjd]
        to_json = {a:b for a,b in zip(attr,vals)}
        requests.post('http://127.0.0.1:8050/test', json=to_json)
        temp += 1
    except ValueError:
        print("bruh")
        pass


ids = requests.get('http://127.0.0.1:8050/get-snr/500')


ids.json()


names = []
for star in star_raw:
    names.append(star[1])


idx = []
for app in list_for_db:
    ids = app[0]
    idx.append(names.index(ids))


idx = list(dict.fromkeys(idx))


dirs = []
fnames = {}
apps_ids = []
for i in idx:
    fname = star_raw[i][0]
    appogeeid = star_raw[i][1]
    apps_ids.append(appogeeid)
    fnames[appogeeid] = fname
    directory = star_raw[i][2]
    lst = directory.split(".")[0:2]
    strng = "/" + "/".join(lst) + "/" + fname
    dirs.append(strng)


aspcapstar_dirs = ["https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec" + d for d in dirs]


dirs = ["https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/" + d for d in dirs]


f = open("data_apstar.txt", "r")
read = f.readlines()
old_list = []
for r in read:
    old_list.append(r.replace("\n", ""))
f.close()


f = open("data_apstar_temp.txt", "a")
for d in dirs:
    f.write(d + "\n")
f.close()


new_dload = [d for d in dirs if d not in old_list]


f = open("data_apstar_download.txt", "a")
for d in new_dload:
    f.write(d + "\n")
f.close()


app_ids = list(fnames.keys())


app_ids.index("2M00212698+0309415")


for i in tqdm(app_ids):
    if fnames[i] == "":
        continue
    try:
        star = fits.open("/Users/bmason3/astrodata/data_apstar/" + fnames[i])
    except FileNotFoundError:
        continue
    apogeeID = i
    
    visits = star[0].header["NVISITS"]
    lst_ccf = []
    lst_mjd = []
    for i in range(0, visits):
        mjd = int(np.floor(star[9].data[i]["jd"]))
        ccf_y = [float(a) if not np.isnan(a) else 0 for a in star[9].data[i]['CCF']]
        lst_ccf.append(ccf_y)
        lst_mjd.append(mjd)
    attr = ["apogeeID", "nvisits", "mjd", "ccf"]
    vals = [apogeeID, visits, lst_mjd, lst_ccf]
    to_json = {a:b for a,b in zip(attr,vals)}
    requests.post('http://127.0.0.1:8050/addccf', json=to_json)


ids = requests.get('http://127.0.0.1:8050/get-ccf/2M14315024+5101159')
ids.json().keys()


temp = [a.replace("apStar","aspcapStar") for a in aspcapstar_dirs]


temp_temp = [a.replace("asStar","aspcapStar") for a in temp]


f = open("data_aspcapstar.txt", "a")
for d in temp_temp:
    f.write(d + "\n")
f.close()


aspcapstar_fnames = [a.split("/")[-1] for a in temp_temp]


tempstar = aspcapstar_fnames[0]
star = fits.open("/Users/blaine/Documents/astrodata/data_aspcapstar/" + tempstar)
y = star[3].data
y = [float(a) if a get_ipython().getoutput("= 0 else 1 for a in y]")
a = np.power(10,star[3].header["CRVAL1"])
step = np.power(10,star[3].header["CDELT1"])
temp= np.arange(star[3].header["CRVAL1"], star[3].header["CRVAL1"]+8575*star[3].header["CDELT1"],star[3].header["CDELT1"])
lam = np.power(10, temp)
plt.plot(lam, y)


[float(a) if not np.isnan(a) else 0 for a in y]
