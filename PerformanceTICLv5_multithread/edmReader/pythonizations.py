""" Pythonizations for reading CMS edm files with FWLite """
import cppyy
import itertools

def repr_helper(*fctNames:list[str]):
    return lambda self: f"{self.__class__.__cpp_name__}(" + ",".join(map(lambda fctName: f"{fctName}={getattr(self, fctName)()}", fctNames)) + ")"

def enum_to_string(enum_class, enum_value):
    for name in dir(enum_class):
        if getattr(enum_class, name) == enum_value:
            return name
        #if not name.startswith("__") and name not in dir(int):
    return enum_class.__name__ + "(UnknownValue)"

def pythonize_std(klass, name):
    if name.startswith("vector"):
        klass.__repr__ = lambda self : self.__class__.__cpp_name__ + "[" + ",".join(itertools.islice((repr(x) for x in self), 20)) + "]"
    elif name.startswith("pair"):
        klass.__repr__ = lambda self : f"pair({repr(self.first)},{repr(self.second)})"
cppyy.py.add_pythonization(pythonize_std, 'std')

def pythonize_root_math(klass, name):
    if name.startswith("PositionVector3D"):
        klass.__repr__ = repr_helper("eta", "phi")
cppyy.py.add_pythonization(pythonize_root_math, 'ROOT::Math')

def pythonize_edm(klass, name):
    if name == "ProductID":
        klass.__repr__ = klass.__str__
    elif name == "OwnVector":
        klass.__repr__ = lambda self : self.__class__.__cpp_name__ + "[" + ",".join(itertools.islice((repr(x) for x in self), 20)) + "]"
    elif name.startswith("Ref"):
        klass.__repr__ = repr_helper("id", "key")
    elif name.startswith("Ptr"):
        def repr_ptr(self):
            try:
                return repr(self.get())
            except:
                return f"{klass.__cpp_name__}(invalid)"
        klass.__repr__ = repr_ptr
    elif name == "EventID":
        klass.__repr__ = repr_helper("run", "luminosityBlock", "event")
cppyy.py.add_pythonization(pythonize_edm, 'edm')

def pythonize_ticl(klass, name):
    if name == "Trackster":
        #klass.__repr__ = lambda self: f"{self.__class__.__cpp_name__}(raw_energy={self.raw_energy()})"
        klass.__repr__ = repr_helper("raw_energy", "regressed_energy")

cppyy.py.add_pythonization(pythonize_ticl, 'ticl')

def pythonize_reco(klass, name):
    if name == "GenParticle":
        klass.__repr__ = repr_helper("energy", "charge", "eta", "pdgId", "isHardProcess")
    elif name == "ElectronSeed":
        def repr_electronSeed(self):
            str = "ElectronSeed("
            if self.isEcalDriven():
                str += "Calo"
            if self.isTrackerDriven():
                str += "Tracker"
            str += f"Driven,charge={self.getCharge()},"
            try:
                str += f"CaloClusterEnergy={self.caloCluster().energy()},CaloClusterEta={self.caloCluster().eta()}"
            except: pass
            return str + ")"
        klass.__repr__ = repr_electronSeed
    elif name == "SuperCluster":
        klass.__repr__ = repr_helper("energy", "rawEnergy", "correctedEnergy", "clustersSize", "eta", "algo", "caloID", "seed")
    elif name == "PFCluster":
        klass.__repr__ = repr_helper("energy", "correctedEnergy", "eta", "layer", "depth", "pt", "algo", "caloID", "seed")
    elif name == "CaloCluster":
        klass.__repr__ = repr_helper("energy", "correctedEnergy", "eta", "algo", "caloID", "seed")
    elif name == "PFRecHit":
        klass.__repr__ = repr_helper("energy", "detId", "layer", "depth", "flags")
    elif name == "CaloID":
        def repr_caloId(self):
            dets = [name for name in dir(klass.Detectors) if not name.startswith("__") and name not in dir(int) and self.detector(getattr(klass.Detectors, name))]
            return ",".join(dets)
        klass.__repr__ = repr_caloId
    elif name == "Track":
        klass.__repr__ = repr_helper("outerEta", "outerPt", "seedDirection")
    elif name == "GsfTrack":
        klass.__repr__ = repr_helper("etaMode", "ptMode", "qoverpMode", "outerEta", "outerPt", "seedDirection")
    elif name == "GsfElectron":
        klass.__repr__ = repr_helper("charge", "caloEnergy", "correctedEcalEnergy", "eSuperClusterOverP")
    elif name in ["PFBlock"]:
        # use operator<< for repr
        klass.__repr__ = klass.__str__

cppyy.py.add_pythonization(pythonize_reco, 'reco')

def pythonize_global(klass, name):
    if name == "DetId":
        klass.__repr__ = lambda self: f"{self.__class__.__cpp_name__}(det={enum_to_string(klass.Detector, self.det())},subdetId={self.subdetId()})"
    elif name == "CaloParticle":
        klass.__repr__ = repr_helper("pdgId", "energy")
cppyy.py.add_pythonization(pythonize_global, '')