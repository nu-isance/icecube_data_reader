from icecube_data_reader.irf.effective_area import IceTrackDR2EffectiveArea
from icecube_data_reader.event_types import IC86


def test_load():
    aeff = IceTrackDR2EffectiveArea.load(IC86)
