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
        temp = list(locate(visit_id, lambda x:x[1] == dat_mjd and x[0] == dat_id))
        if len(temp) > 0:
            ret.append(visit[temp[0]][0])
        else:
            continue
    return ret

def dict_display(dict,N):
    for i in range(0,N):
        print(list(dict.items())[i])
        print("")

def main():
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
    #Get the index of all_visit data that matches sb2_appid_lst
    filtered_allVisit = filter_allVisit(visit_id_list, visit_raw, sb2_info_list)
    f = open("dat_allvisit.txt", "a")
    for x in filtered_allVisit:
        f.write(x)
    f.close()
    return 0
if __name__ == "__main__":
    main()