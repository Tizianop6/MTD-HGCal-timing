import uproot
import numpy as np



class DQMReader:
    """ Reads DQM files, provides some convenience functions """
    class EGammaV:
        def __init__(self, folder):
            self.folder = folder
            self.electronValidator = self.folder["ElectronMcSignalValidator"]

        @property
        def absEtaEff(self):
            return self.electronValidator["h_ele_absetaEff_Extended"]
        @property
        def ptEff(self):
            return self.electronValidator["h_ele_ptEff"]
        
        def efficiency(self, mode="pt"):
            """ mode should be all the same """
            if mode == "pt":
                return self.electronValidator["h_mc_Pt_matched"].to_hist().sum(flow=True) / self.electronValidator["h_mc_Pt"].to_hist().sum(flow=True)
            elif mode == "abseta":
                return self.electronValidator["h_mc_AbsEta_Extended_matched"].to_hist().sum(flow=True) / self.electronValidator["h_mc_AbsEta_Extended"].to_hist().sum(flow=True)
            elif mode == "eta":
                return self.electronValidator["h_mc_Eta_Extended_matched"].to_hist().sum(flow=True) / self.electronValidator["h_mc_Eta_Extended"].to_hist().sum(flow=True)

    def __init__(self, path:str):
        self.file = uproot.open(path)
        try:
            self.egammaV = self.EGammaV(self.file["DQMData"]["Run 1"]["EgammaV"]["Run summary"])
        except KeyError: pass
    
    def egammaVForWP(self, wp:float) -> EGammaV:
        return self.EGammaV(self.file["DQMData"]["Run 1"][f"EgammaVDNNWP{str(float(wp)).replace('.', 'p')}"]["Run summary"])
