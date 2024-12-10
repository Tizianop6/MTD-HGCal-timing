""" Tools to read simToReco associations (ie SimTrackster -> trackster) from TICLDumper output """
import pandas as pd

from .tracksters import supercluster_joinTracksters



def superclusterToSim_df(supercluster_df:pd.DataFrame, assocs_bestScore_recoToSim_df:pd.DataFrame, tracksters_df:pd.DataFrame) -> pd.DataFrame:
    """ Make Df of tracksters in superclusters joined with recoToSim associations and trackster information
    
    Index : eventInternal	supercls_id	ts_in_supercls_id	
    Columns : ts_id	simts_id	score	sharedE	raw_energy	regressed_energy"""
    df = supercluster_df.join(assocs_bestScore_recoToSim_df(), on=["eventInternal", "ts_id"])
    df.score = df.score.fillna(1)
    df.sharedE = df.sharedE.fillna(0)
    
    return supercluster_joinTracksters(df, tracksters_df)