from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from sklearn.decomposition import FastICA
import requests
from tqdm import tqdm


sb2_fits = fits.open("apogee_sb2s-v1_0.fits", memmap=True)
sb2_data = sb2_fits[1].data
all_visit = fits.open("allVisit-dr17-synspec_rev1.fits")
all_visit_data = all_visit[1].data
N_all_visit = all_visit[1].data.shape[0]


sb2_filter = []
for i in range(0, sb2_data.shape[0]):
    arr = sb2_data[i][10]
    if arr[0] >= 3 and arr[1] >= 3 and arr[2] == 0 and arr[3] == 0:
        sb2_filter.append(i)


filtered_sb2_data = []
for i in range(0,len(sb2_filter)):
    filtered_sb2_data.append(sb2_data[sb2_filter[i]])


all_sb2_names = []
N = len(filtered_sb2_data)
for i in range(0, N):
    all_sb2_names.append(filtered_sb2_data[i][0])


res = []
[res.append(x) for x in all_sb2_names if x not in res];


res


lst_of_all_data = []
for j in tqdm(range(0, N_all_visit)):
    lst_of_all_data.append(all_visit_data[j][0])


from more_itertools import locate
ID = res[0]
temp = list(locate(lst_of_all_data, lambda x: x == ID))
print(temp)


binary = []
for i in tqdm(range(0, len(res))):
    ID = res[i]
    ind = 0
    temp = list(locate(lst_of_all_data, lambda x: x == ID))
    for indxs in temp:
        binary.append(all_visit_data[indxs])


all_visit_mjd = [binary[i][7] for i in range(len(binary))]


all_sb2_mjd = list(set([int(np.floor(filtered_sb2_data[i][1])) for i in range(len(filtered_sb2_data))]))


#NEED TO COLLECT NEW DATA THAT MATCHES MJD OF TESTING TO MJD OF BINARY
temp_list = []
appp_id = []
for i in tqdm(range(0,len(all_sb2_mjd))):
    try:
        mjd_0 = all_sb2_mjd[i]
        idx = all_visit_mjd.index(mjd_0)
        temp_list.append(binary[idx][3])
        appp_id.append(binary[idx][0])
    except ValueError:
        pass


len(temp_list)


appp_id


temp_list = list(set(temp_list))


f = open("data.txt", "a")
for x in temp_list:
    one = x.split("-")[2]
    two = x.split("-")[3]
    f.write("https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/plates/" + one + "/" + two + "/" + x + "\n")
f.close()


all_sb2_mjd = list(set(all_sb2_mjd))


for i in range(len(filtered_sb2_data)):
    if filtered_sb2_data[i][10][0] == 0 and filtered_sb2_data[i][10][1] == 0:
        print(filtered_sb2_data[i][10])


start = 0
for i in tqdm(range(0,len(temp_list))):
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


ids = requests.get('http://127.0.0.1:8050/get-snr/50')


ids.json()


temp = dict(sorted(ids.json().items(), key=lambda item: item[1]))


[i[1][0] + "  " + str(i[1][1]) for i in temp.items()]


binary_star_vist = requests.get('http://127.0.0.1:8050/get-binary/2M00023036+8524194')


temp = binary_star_vist.json()


temp.keys()


res = temp["2M00023036+8524194102.37000274658203"]
df = pd.DataFrame.from_dict(res)


df


i=0
data_y = df["A"][0]
data_x = df["A"][1]
plt.plot(data_x, data_y)


ica_test = np.vstack((data_x,data_y))


ica = FastICA(n_components=2)
S_ica_ = ica.fit_transform(ica_test.T) # Estimate the sources
plt.plot(data_x, S_ica_[:,0])
plt.plot(data_x, S_ica_[:,1])


allstar = fits.open("../../../Downloads/allStar-dr17-synspec_rev1.fits")


allstarLen = len(allstar[1].data)
names = []
for i in range(allstarLen):
    names.append(allstar[1].data[i][1])


appp_id = list(set(appp_id))


idxList = []
for i in tqdm(range(0, len(appp_id))):
    ids = appp_id[i]
    idxList.append(names.index(ids))


idxList = list(set(idxList))


allstar[1].data[idx][1]


dirs = []
fnames = {}
apps_ids = []
for idx in idxList:
    fname = allstar[1].data[idx][0]
    appogeeid = allstar[1].data[idx][1]
    apps_ids.append(appogeeid)
    fnames[appogeeid] = fname
    directory = allstar[1].data[idx][2]
    lst = directory.split(".")[0:2]
    strng = "/" + "/".join(lst) + "/" + fname
    dirs.append(strng)


dirs = ["https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/" + d for d in dirs]


f = open("data_apstar.txt", "a")
for d in dirs:
    f.write(d + "\n")
f.close()


data_for_apstar = requests.get('http://127.0.0.1:8050/get-binary/2M00362511+3408151')


star = fits.open("../../../data_apstar/apStar-dr17-2M18453580-0240470.fits")


for i in apps_ids:
    print(i)


for i in tqdm(apps_ids):
    if fnames[i] == "":
        continue
    try:
        star = fits.open("../../../data_apstar/" + fnames[i])
    except FileNotFoundError:
        continue
    apogeeID = i
    visits = star[0].header["NVISITS"]
    lst_ccf = []
    lst_mjd = []
    for i in range(2, visits):
        mjd = int(np.floor(star[9].data[i]["jd"]))
        ccf_y = [float(a) if not np.isnan(a) else 0 for a in star[9].data[i]['CCF']]
        lst_ccf.append(ccf_y)
        lst_mjd.append(mjd)
    attr = ["apogeeID", "nvisits", "mjd", "ccf"]
    vals = [apogeeID, visits, lst_mjd, lst_ccf]
    to_json = {a:b for a,b in zip(attr,vals)}
    requests.post('http://127.0.0.1:8050/addccf', json=to_json)


#cross cor
star[9].data[0]["jd"]


visits = star[0].header["NVISITS"]
for i in range(2,visits):
    ccf_y = star[9].data[i]['CCF']
    ccf_x = np.arange(-382,383, 1)
    plt.xlim(-100,100)
    plt.plot(ccf_x, ccf_y)


temp = requests.get('http://127.0.0.1:8050/get-binary/2M00004521-7219055')


temp.json()["2M00004521-7219055114.77999877929688"]["MJD"]


ids = requests.get('http://127.0.0.1:8050/get-ccf/2M00004521-7219055')
ids.json().keys()


data_for_apstar = requests.get('http://127.0.0.1:8050/get-binary/2M00004521-7219055')


data_for_apstar.json()["2M00004521-7219055114.77999877929688"]["MJD"]


data_for_apstar.json()["2M00004521-7219055114.77999877929688"]["fname"]


testing_data = fits.open("../../../data/apVisit-dr17-8702-57348-257.fits")


testing_data[0].header["MJD5"]


testing_data[0].header["RA"]


testing_data[0].header["DEC"]


testing_data[0].header["OBJID"]


star = fits.open("../../../data_apstar/apStar-dr17-2M00212698+0309415.fits")


int(np.floor(star[9].data[0]["jd"]))-2400000


result = dict(sorted(ids.json().items(), key=lambda item: item[1]))
for i in result.items():
    if i[1][0] == 
[i[1][0] for i in result.items()]


a = np.power(10,star[0].header["CRVAL1"])


step = np.power(10,star[0].header["CDELT1"])


temp= np.arange(star[0].header["CRVAL1"], star[0].header["CRVAL1"]+8575*star[0].header["CDELT1"],star[0].header["CDELT1"])


lam = np.power(10, temp)


plt.plot(lam, y)


nm
