from analyzer.driver.computations import DataframeComputation
from analyzer.driver.fileTools import SingleInputReader, SingleInputReaderFWLite
from analyzer.dumperReader.reader import DumperReader, getCPToSuperclusterProperties
import awkward as ak

# cannot use a lambda as multirprocessing does not work due to pickle issues
def _CPToSuperclusterProperties_fct(reader:SingleInputReader):
    reader = reader.ticlDumperReader
    return getCPToSuperclusterProperties(reader.supercluster_all_df, reader.assocs_bestScore_simToReco_df, 
        reader.tracksters_zipped[["ts_id", "raw_energy", "raw_em_energy", "regressed_energy", "raw_pt", "barycenter_eta"]], reader.simTrackstersCP_df)
CPToSuperclusterProperties = DataframeComputation(_CPToSuperclusterProperties_fct, "CPToSupercluster")

def _CPToSuperclusterProperties_recoSC_fct(reader:SingleInputReaderFWLite):
    return getCPToSuperclusterProperties(ak.to_dataframe(reader.ticlDumperReader.fileDir["superclustering/recoSuperCluster_constituentTs"].array(), anonymous="ts_id",
            levelname=lambda x : {0:"eventInternal", 1:"supercls_id", 2:"ts_in_supercls_id"}[x]),
        reader.ticlDumperReader.assocs_bestScore_simToReco_df, 
        reader.ticlDumperReader.tracksters_zipped[["ts_id", "raw_energy", "raw_em_energy", "regressed_energy", "raw_pt", "barycenter_eta"]], reader.ticlDumperReader.simTrackstersCP_df)
CPToSuperclusterProperties_recoSC = DataframeComputation(_CPToSuperclusterProperties_recoSC_fct, "CPToSupercluster_recoSC")

def _CPToSuperclusterProperties_fwlite_fct(reader:SingleInputReaderFWLite):
    return getCPToSuperclusterProperties(reader.fwliteDataframesReader.supercls, reader.ticlDumperReader.assocs_bestScore_simToReco_df, 
        reader.ticlDumperReader.tracksters_zipped[["ts_id", "raw_energy", "raw_em_energy", "regressed_energy", "raw_pt", "barycenter_eta"]], reader.ticlDumperReader.simTrackstersCP_df)
CPToSuperclusterProperties_fwlite = DataframeComputation(_CPToSuperclusterProperties_fwlite_fct, "CPToSupercluster_fwlite")
