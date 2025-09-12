###################################################################################################
# Setup
# ---------------------
#
# Imports
from eegnb import generate_save_fn
from eegnb.devices.eeg import EEG
from eegnb.experiments import AuditoryOddball, VisualSSVEP


def main():
    """Entry point for running the Visual N170 experiment.

    Wrapped in a function and guarded by __main__ to be safe with
    multiprocessing on Windows (spawn), preventing duplicate runs.
    """
    # Define some variables
    board_name = "lsl"  # board name
    experiment_name = "audio_oddball"  # experiment name
    subject_id = 1  # test subject id
    session_nb = 0  # session number
    record_duration = 120  # recording duration (short for quick test)

    # generate save path
    save_fn = generate_save_fn(
        board_name, experiment_name, subject_id, session_nb)

    # create device object
    eeg_device = EEG(device=board_name)

    # Experiment type
    experiment = AuditoryOddball(eeg=eeg_device, save_fn=save_fn)

    ###################################################################################################
    # Run experiment
    # ---------------------
    #
    print(f"Saving to: {save_fn}")
    experiment.run()

    # Saved csv location
    print("Recording saved in", experiment.save_fn)


if __name__ == "__main__":
    # On Windows, multiprocessing uses 'spawn' and re-imports this module.
    # The __main__ guard prevents the experiment from running twice.
    main()
