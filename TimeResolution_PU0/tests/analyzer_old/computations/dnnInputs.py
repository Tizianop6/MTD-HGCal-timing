import math
import hist
import awkward as ak

from analyzer.driver.fileTools import SingleInputReader
from analyzer.driver.computations import HistogramComputation, DataframeComputation
from analyzer.dumperReader.dnnSampleReader import DNNSampleReader, seedToDataframe




def _fillFctFeature(h:hist.Hist, reader:DNNSampleReader, featureName:str):
    pass

def makeComputationForFeature(feat:str) -> HistogramComputation:
    return HistogramComputation(
        hist.Hist(feature_axises[feat],
             name=feat), _fillFctFeature, paramsForFillFct=dict(featureName=feat, ))


# cannot use a lambda as multirprocessing does not work due to pickle issues
def _highestPtSeedProperties(reader:SingleInputReader):
    return seedToDataframe(reader.dnnSampleDumperReader.bestSeedProperties)
seed_highestPt = DataframeComputation(_highestPtSeedProperties, "seed_highestPt")

def _highestPtSeed_pairsProperties(reader:SingleInputReader):
    return ak.to_dataframe(reader.dnnSampleDumperReader.pairsWithHighestPtSeed, levelname=lambda x : {0:"eventInternal", 1:"pairInternal"}[x])
pairs_highestPtSeed = DataframeComputation(_highestPtSeed_pairsProperties, "pairs_highestPtSeed")
