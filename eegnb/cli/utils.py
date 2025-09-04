
#change the pref libraty to PTB and set the latency mode to high precision
from psychopy import prefs
prefs.hardware['audioLib'] = 'PTB'
prefs.hardware['audioLatencyMode'] = 3


from eegnb.devices.eeg import EEG

from eegnb.experiments import VisualN170, Experiment
from eegnb.experiments import VisualP300
from eegnb.experiments import VisualSSVEP
# AuditoryOddball is unavailable on macOS Apple Silicon due to PTB limitations.
try:
    from eegnb.experiments import AuditoryOddball
    _HAVE_AOB = True
except Exception:
    AuditoryOddball = None  # type: ignore
    _HAVE_AOB = False
from eegnb.experiments.visual_cueing import cueing
from eegnb.experiments.visual_codeprose import codeprose

# Lazy/guarded imports for audio experiments to avoid hard dependency on
# PsychToolbox or other audio backends unless the user selects them.
try:
    from eegnb.experiments.auditory_oddball import diaconescu  # imports aMMN -> sound
    _HAVE_DIAC = True
except Exception:
    diaconescu = None  # type: ignore
    _HAVE_DIAC = False

try:
    from eegnb.experiments.auditory_ssaep import ssaep, ssaep_onefreq
    _HAVE_SSAEP = True
except Exception:
    ssaep = None  # type: ignore
    ssaep_onefreq = None  # type: ignore
    _HAVE_SSAEP = False
from typing import Optional


# New Experiment Class structure has a different initilization, to be noted
experiments = {
    "visual-N170": VisualN170(),
    "visual-P300": VisualP300(),
    "visual-SSVEP": VisualSSVEP(),
    "visual-cue": cueing,
    "visual-codeprose": codeprose,
}

if _HAVE_SSAEP:
    experiments["auditory-SSAEP orig"] = ssaep
    experiments["auditory-SSAEP onefreq"] = ssaep_onefreq

if _HAVE_DIAC:
    experiments["auditory-oddball diaconescu"] = diaconescu

if _HAVE_AOB:
    experiments["auditory-oddball orig"] = AuditoryOddball()


def get_exp_desc(exp: str):
    if exp in experiments:
        module = experiments[exp]
        if hasattr(module, "__title__"):
            return module.__title__  # type: ignore
    return "{} (no description)".format(exp)


def run_experiment(
    experiment: str, eeg_device: EEG, record_duration: Optional[float] = None, save_fn=None
):
    if experiment in experiments:
        module = experiments[experiment]

        # Condition added for different run types of old and new experiment class structure
        if isinstance(module, Experiment.BaseExperiment):
            module.duration = record_duration
            module.eeg = eeg_device
            module.save_fn = save_fn
            module.run()
        else:
            module.present(duration=record_duration, eeg=eeg_device, save_fn=save_fn)  # type: ignore
    else:
        print("\nError: Unknown experiment '{}'".format(experiment))
        print("\nExperiment can be one of:")
        print("\n".join([" - " + exp for exp in experiments]))
