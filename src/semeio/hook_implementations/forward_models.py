import ert

from semeio.forward_models import (
    OTS,
    Design2Params,
    DesignKW,
    GenDataRFT,
    InsertNoSim,
    Pyscal,
    RemoveNoSim,
    ReplaceString,
)


@ert.plugin(name="semeio")
def installable_forward_model_steps():
    return [
        Design2Params,
        DesignKW,
        GenDataRFT,
        OTS,
        Pyscal,
        InsertNoSim,
        RemoveNoSim,
        ReplaceString,
    ]
