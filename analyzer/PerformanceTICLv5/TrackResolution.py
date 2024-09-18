import sys
sys.path.append("../..")
from functools import partial
from typing import Literal

import uproot
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import math
import mplhep as hep
plt.style.use(hep.style.CMS)
hep.cms.label("Private work (CMS simulation)", loc=2)

import hist
import ROOT
from analyzer.dumperReader.reader import *
from analyzer.driver.fileTools import *
from analyzer.driver.computations import *
from analyzer.computations.tracksters import tracksters_seedProperties, CPtoTrackster_properties, CPtoTracksterMerged_properties
from analyzer.energy_resolution.fit import *
import os
from matplotlib.colors import ListedColormap
from matplotlib import cm
from utilities import *
from tqdm import tqdm
import os, ROOT
import cmsstyle as CMS
import multiprocessing


CMS.SetExtraText("Private work (CMS simulation)")
CMS.SetLumi("")

from numba import prange,njit

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    return directory_path


fileV5 = "/eos/user/t/tipaulet/Local_Energy_Samples/SinglePionTiming_1p9_50GeV/histo/"
# fileV4 = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv5Performance/TimeResolution/SinglePionTiming_2p2_100GeV/histo/"

OutputDir = "/eos/user/t/tipaulet/www/test_resolution/"

create_directory(OutputDir)

#@njit(parallel=True)
def ReadFileAndFit(filename,return_dict,key,tag="prova",eta=1.9, maxfiles=1):


    dumperInputV5 = DumperInputManager([
        filename
    ], limitFileCount=maxfiles)

    denominator = []
    numerator = []


    ticl_residuals={"avg":[],
                    "hgcal":[],
                    "mtd":[]}
    residual_list=[]
    for i in range(len(dumperInputV5.inputReaders)):
        dumper = dumperInputV5.inputReaders[i].ticlDumperReader
        tms = dumper.trackstersMerged
        cands = dumper.candidates
        simCands = dumper.simCandidates
        ass = dumper.associations
        sims = dumper.simTrackstersCP
        simtrackstersSC = dumper.simTrackstersSC
        
        tracks = dumper.tracks
        #print("file ", i)

        for ev in range(len(tms)):
            tmEv = tms[ev]
            candEv = cands[ev]
            simCandEv = simCands[ev]
            assEv = ass[ev]
            simEv = sims[ev]
            tracksEv = tracks[ev]
            tsSCEv = simtrackstersSC[ev]
            for simCand_idx in range(len(tsSCEv.barycenter_x)):

                #simRawEnergy = simCandEv.simTICLCandidate_raw_energy[simCand_idx]
                simRegrPt = tsSCEv.regressed_pt[simCand_idx]
                #simTime = simCandEv.simTICLCandidate_time[simCand_idx]
                simTrack = tsSCEv.trackIdx[simCand_idx]



                tk_idx_in_coll = -1
                try:
                    tk_idx_in_coll = np.where(tracksEv.track_id == simTrack)[0][0] 
                except:
                    continue
                if tk_idx_in_coll == -1:
                    continue
                #if tk_idx_in_coll == -1 or tracksEv.track_pt[tk_idx_in_coll] < 1 or tracksEv.track_missing_outer_hits[tk_idx_in_coll] > 5 or not tracksEv.track_quality[tk_idx_in_coll]: 
                #    continue
                

                residual=tracksEv.track_pt[tk_idx_in_coll]/simRegrPt#*math.cosh(eta)
                residual_list.append(residual)
                
    print(len(residual_list),key)


    plt.figure()
    #print(ticl_residuals)
    if(key>30):
        rng=8.5
    else:
        rng=2.5
    y_h,x_h,_=plt.hist(residual_list, range=(-rng,+rng), bins = 100, histtype = "step", lw = 2, color = "xkcd:magenta", label="Combination")
    #plt.hist(ticl_residuals["mtd"], range=(-0.1,+0.1), bins = 100, histtype = "step", lw = 2, color = "xkcd:azure", label="ETL")
    #plt.hist(ticl_residuals["hgcal"], range=(-0.1,+0.1), bins = 100, histtype = "step", lw = 2, color = 'xkcd:green', label="HGCAL")

    xlist=np.linspace(-rng,rng,1000)

    hlist=[]
    hlist.append(hist.Hist(hist.axis.Regular(100,-rng,rng,name="tres"), name="name", label="label"))
    hlist[0].fill(residual_list)


    #print(fitMultiHistogram(hlist))
    fitres=fitMultiHistogram(hlist)
    #print(fitres)

    cruj=partial(cruijff, A=fitres[0][0].params.A, m=fitres[0][0].params.m, sigmaL=fitres[0][0].params.sigmaL,
        sigmaR=fitres[0][0].params.sigmaR, alphaL=fitres[0][0].params.alphaL, alphaR=fitres[0][0].params.alphaR)

    plt.plot(xlist,cruj(np.array(xlist)),linewidth=4,color = "xkcd:azure")
    #print(cruj(np.array(xlist)))



    #print(fitres[0][0].params.A,fitres[0][1].params.A,fitres[0][2].params.A)
    plt.legend()
    plt.xlabel(("$p_T$ residual [GeV]"))
    plt.ylabel("Entries")
    hep.cms.text("Private work (Simulation)", loc=0)                                                  
    
    match = re.search(r'(\d+)GeV', tag)
    
    plt.text(0.045, y_h.max()*0.4, "$p_T="+match.group(1)+"$ GeV\n$\eta="+str(eta)+"$")

    plt.savefig(OutputDir + tag+".png")

    
    res=(fitres[0][0].params.sigmaL+fitres[0][0].params.sigmaR)/2
    cov_matrix=fitres[0][0].covMatrix
    resErr=math.sqrt(0.25*(cov_matrix[2][2]+cov_matrix[3][3])+ 0.5*cov_matrix[2][3])
    resAndErrs=[res,resErr]

    
    return_dict[key]=resAndErrs
    
    #return (resAndErrs)

#plot_ratio_single(numerator, denominator, 10, [1,200], label1="TICLv5", xlabel="Sim Regressend Energy [GeV]", saveFileName=OutputDir + "trackEff_v5.png")


def PlotRes(dict_resolutions,tag="test",eta=1.9):


    #print(dict_resolutions)
    #print(dict_resolutions[2])
    counter=0
    pts={'x':[],"y":[],"yErr":[],"y2":[],"y2Err":[]}
    for enkey in dict_resolutions.keys():
        #pts["x"].append(enkey*math.cosh(eta))
        pts["x"].append(enkey*math.cosh(eta))
        pts["y"].append(dict_resolutions[enkey][0]*math.cosh(eta))
        pts["yErr"].append(dict_resolutions[enkey][1]*math.cosh(eta))
        pts["y2"].append(dict_resolutions[enkey][0]*math.cosh(eta)/(enkey*math.cosh(eta)))
        pts["y2Err"].append(dict_resolutions[enkey][1]*math.cosh(eta)/(enkey*math.cosh(eta)))
        
        
    plt.clf()
    plt.errorbar(pts["x"],pts["y"],pts["yErr"],label="Combination",color="xkcd:magenta",marker="^",linestyle='',capsize=3,capthick=2)
    #plt.ylim(0, 0.04)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    plt.text(60, 0.002, "Single pion 0PU\n$\eta="+str(eta)+"$")
    plt.legend()

    plt.xlabel("$Energy$ [GeV]")
    plt.ylabel("Sigma [GeV]")
    plt.savefig(OutputDir + tag+"eta"+str(eta)+"Sigma.png")


    plt.clf()
    
    plt.errorbar(pts["x"],pts["y2"],pts["y2Err"],label="Combination",color="xkcd:magenta",marker="^",linestyle='',capsize=3,capthick=2)
    
    #plt.ylim(0, 0.04)
    hep.cms.text("Simulation Preliminary", loc=0)                                                  
    plt.text(60, 0.002, "Single pion 0PU\n$\eta="+str(eta)+"$")
    plt.legend()

    plt.xlabel("$Energy$ [GeV]")
    plt.ylabel("Resolution ")
    plt.savefig(OutputDir + tag+"eta"+str(eta)+"SigmaEdE.png")








#print(ReadFileAndFit(fileV5))


#folders_1p9 = ["SinglePionTiming_1p9_100GeV", "SinglePionTiming_1p9_10GeV"]


if __name__ == "__main__":


    folders_1p9 = ["SinglePionTiming_1p9_100GeV", "SinglePionTiming_1p9_10GeV",
     "SinglePionTiming_1p9_15GeV", "SinglePionTiming_1p9_2GeV",
     "SinglePionTiming_1p9_30GeV", "SinglePionTiming_1p9_4GeV",
     "SinglePionTiming_1p9_50GeV", "SinglePionTiming_1p9_6GeV",
     "SinglePionTiming_1p9_8GeV"]


    dict_resolutions={}

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = [] 

    eta=1.9
    maxfiles=50
    

    for i in tqdm(range(len(folders_1p9))):
        file_name="/eos/user/t/tipaulet/Local_Energy_Samples/"+folders_1p9[i]+"/histo/"
        match = re.search(r'(\d+)GeV', folders_1p9[i])

        p = multiprocessing.Process(target=ReadFileAndFit, args=(file_name,return_dict,int(match.group(1)),folders_1p9[i],eta,maxfiles))
        jobs.append(p)
        p.start()

        
    for proc in jobs:
        proc.join()
    
    print(return_dict)

    dict_resolutions=return_dict

    PlotRes(dict_resolutions,tag="pT",eta=eta)


    '''
    dict_resolutions={}    
    folders_2p2 = ["SinglePionTiming_2p2_100GeV", "SinglePionTiming_2p2_10GeV",
     "SinglePionTiming_2p2_15GeV", "SinglePionTiming_2p2_2GeV",
     "SinglePionTiming_2p2_30GeV", "SinglePionTiming_2p2_4GeV",
     "SinglePionTiming_2p2_50GeV", "SinglePionTiming_2p2_6GeV",
     "SinglePionTiming_2p2_8GeV"]




    eta=2.2




    PlotRes(dict_resolutions,tag="pT",eta=eta)
    '''
