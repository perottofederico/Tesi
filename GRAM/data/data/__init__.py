
from .EEGDataset import EEGDataset
#from .THINGSDatasetOld import THINGSDatasetOld
from .THINGSDataset import THINGSDataset
from .THINGSDatasetCrossAtt import THINGSDatasetCrossAtt

data_registry={
                 'eeg': EEGDataset,
                 'THINGS': THINGSDataset,
                 'THINGS_crossatt': THINGSDatasetCrossAtt,
                 }
