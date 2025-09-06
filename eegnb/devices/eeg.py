""" Abstraction for the various supported EEG devices.

    1. Determine which backend to use for the board.
    2.

"""

import sys
import time
import logging
from time import sleep
from multiprocessing import Process, Event

import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
from muselsl import stream, list_muses, record, constants as mlsl_cnsts
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

from eegnb.devices.utils import (
    get_openbci_usb,
    create_stim_array,
    SAMPLE_FREQS,
    EEG_INDICES,
    EEG_CHANNELS,
)


logger = logging.getLogger(__name__)

# Helper to extract EEG-only indices and labels from LSL stream info


def _lsl_eeg_indices_and_labels(info) -> tuple[list[int], list[str]]:
    """Return indices and labels for channels with type 'EEG'.

    If no channel types are present in metadata, fall back to all channels.
    If types exist but none are 'EEG', fall back to all channels.
    """
    n_chans = info.channel_count()
    desc = info.desc()
    eeg_indices: list[int] = []
    labels: list[str] = []
    all_types: list[str] = []
    all_labels: list[str] = []

    try:
        ch = desc.child("channels").first_child()
        for i in range(n_chans):
            if i > 0:
                ch = ch.next_sibling()
            lab = ch.child_value("label") if ch else ""
            ctype = ch.child_value("type") if ch else ""
            all_labels.append(lab if lab != "" else f"eeg_{i}")
            all_types.append(ctype)
    except Exception:
        # No metadata; assume all EEG with generic labels
        return list(range(n_chans)), [f"eeg_{i}" for i in range(n_chans)]

    typed_present = any(t and t.strip() for t in all_types)
    if typed_present:
        eeg_indices = [i for i, t in enumerate(
            all_types) if (t or "").strip().lower() == "eeg"]
        if not eeg_indices:
            # types exist but none 'EEG' -> fallback to all
            eeg_indices = list(range(n_chans))
    else:
        eeg_indices = list(range(n_chans))

    labels = [all_labels[i] for i in eeg_indices]
    return eeg_indices, labels

# Dedicated LSL recording worker to preserve channel labels and embed markers


def lsl_record_worker(save_fn: str, stop_event, poll_interval: float = 0.05, startup_timeout: float = 10.0):
    """Record from the first available LSL EEG stream and a 'Markers' stream until stop_event is set.

    Writes a CSV with columns: timestamps, <channel labels...>, stim
    """
    try:
        # Resolve EEG stream and create inlet (retry until available or stop requested)
        eeg_inlet = None
        start_wait = time.time()
        while not stop_event.is_set() and eeg_inlet is None:
            eeg_streams = resolve_byprop("type", "EEG", timeout=1)
            if eeg_streams:
                eeg_inlet = StreamInlet(
                    eeg_streams[0], max_chunklen=mlsl_cnsts.LSL_EEG_CHUNK, recover=True
                )
                break
            if time.time() - start_wait > startup_timeout:
                raise RuntimeError(
                    "No LSL EEG stream found for recording within timeout")

        # Resolve markers stream (optional)
        marker_inlet = None
        try:
            print(f"DEBUG: Looking for marker streams...")
            marker_streams = resolve_byprop("type", "Markers", timeout=5)  # Increased timeout
            if marker_streams:
                marker_inlet = StreamInlet(
                    marker_streams[0], max_chunklen=128, recover=True)
                print(f"DEBUG: Found marker stream: {marker_streams[0].name()}")
            else:
                print("DEBUG: No marker stream found - markers will not be recorded")
        except Exception as e:
            print(f"DEBUG: Failed to resolve marker stream: {e}")
            marker_inlet = None

        # Extract channel labels from EEG stream metadata
        info = eeg_inlet.info()
        eeg_indices, ch_names = _lsl_eeg_indices_and_labels(info)

        # Buffers
        all_ts = []
        all_samples = []
        marker_events = []  # list of (timestamp, value)

        # Get initial timestamps to understand the clock offset
        first_eeg_ts = None
        first_marker_ts = None

        timeout = max(0.01, min(0.25, poll_interval))
        while not stop_event.is_set():
            # Pull EEG chunk
            samples, timestamps = eeg_inlet.pull_chunk(
                timeout=timeout, max_samples=mlsl_cnsts.LSL_EEG_CHUNK)
            if timestamps:
                if first_eeg_ts is None:
                    first_eeg_ts = timestamps[0]
                    print(f"DEBUG: First EEG timestamp: {first_eeg_ts}")
                all_ts.extend(timestamps)
                all_samples.extend(samples)

            # Pull any marker events
            if marker_inlet is not None:
                mvals, m_ts = marker_inlet.pull_chunk(
                    timeout=0.0, max_samples=256)
                if m_ts:
                    # Flatten marker values to ints if provided as lists
                    for v, ts in zip(mvals, m_ts):
                        try:
                            mv = int(v[0]) if isinstance(
                                v, (list, tuple)) else int(v)
                        except Exception:
                            mv = 0
                        if first_marker_ts is None:
                            first_marker_ts = ts
                            print(f"DEBUG: First marker timestamp: {first_marker_ts}")
                        marker_events.append((ts, mv))
                        print(f"DEBUG: Received marker: {mv} at {ts}")
            else:
                # Log once that no marker inlet is available
                if len(all_ts) == 1:  # Log only on first EEG sample
                    print("DEBUG: No marker inlet available - markers will not be recorded")

        # Build DataFrame
        if not all_ts:
            # Nothing captured; still create an empty CSV with headers for traceability
            df_empty = pd.DataFrame(
                columns=["timestamps"] + ch_names + ["stim"])
            df_empty.to_csv(save_fn, index=False)
            return

        eeg_arr = np.asarray(all_samples, dtype=float)
        # Select only EEG channels by indices
        try:
            eeg_arr = eeg_arr[:, eeg_indices]
        except Exception:
            # Shape mismatch fallback: keep as-is
            pass
        ts_arr = np.asarray(all_ts, dtype=float)

        # Compute stim column by assigning the last marker seen up to each timestamp
        stim = np.zeros_like(ts_arr, dtype=int)
        if marker_events:
            print(f"DEBUG: Processing {len(marker_events)} marker events")
            print(f"DEBUG: EEG timestamp range: {ts_arr[0]:.3f} to {ts_arr[-1]:.3f}")
            print(f"DEBUG: Marker timestamp range: {marker_events[0][0]:.3f} to {marker_events[-1][0]:.3f}")
            
            # Calculate clock offset between EEG and marker streams
            # If timestamps are on different scales, we need to synchronize them
            eeg_start = ts_arr[0]
            marker_start = marker_events[0][0]
            
            # Check if timestamps are on vastly different scales (different clocks)
            if abs(eeg_start - marker_start) > 1000:  # More than 1000 seconds difference suggests different clocks
                print(f"DEBUG: Clock offset detected. EEG start: {eeg_start}, Marker start: {marker_start}")
                # Calculate offset and adjust marker timestamps
                offset = eeg_start - marker_start
                print(f"DEBUG: Applying clock offset: {offset}")
                
                # Apply offset to all marker timestamps
                marker_events = [(ts + offset, val) for ts, val in marker_events]
                print(f"DEBUG: Adjusted marker timestamp range: {marker_events[0][0]:.3f} to {marker_events[-1][0]:.3f}")
            
            marker_events.sort(key=lambda x: x[0])
            mi = 0
            last_val = 0
            matches = 0
            for i, ts in enumerate(ts_arr):
                while mi < len(marker_events) and marker_events[mi][0] <= ts:
                    last_val = marker_events[mi][1]
                    matches += 1
                    mi += 1
                stim[i] = last_val
            print(f"DEBUG: Applied {matches} marker timestamps to EEG data")
        else:
            print("DEBUG: No marker events found - stim column will be all zeros")

        data_df = pd.DataFrame(eeg_arr, columns=ch_names)
        data_df.insert(0, "timestamps", ts_arr)
        data_df["stim"] = stim
        data_df.to_csv(save_fn, index=False)
    except Exception as e:
        # Ensure we write a minimal file to signal an attempt even if error happens
        try:
            pd.DataFrame(columns=["timestamps", "stim"]
                         ).to_csv(save_fn, index=False)
        except Exception:
            pass
        logger.exception(f"LSL record worker failed: {e}")


# list of brainflow devices
brainflow_devices = [
    "ganglion",
    "ganglion_wifi",
    "cyton",
    "cyton_wifi",
    "cyton_daisy",
    "cyton_daisy_wifi",
    "brainbit",
    "unicorn",
    "synthetic",
    "brainbit",
    "notion1",
    "notion2",
    "freeeeg32",
    "crown",
    "museS_bfn",  # bfn = brainflow with native bluetooth;
    "museS_bfb",  # bfb = brainflow with BLED dongle bluetooth
    "muse2_bfn",
    "muse2_bfb",
    "muse2016_bfn",
    "muse2016_bfb",
]


class EEG:
    device_name: str
    stream_started: bool = False

    def __init__(
        self,
        device=None,
        serial_port=None,
        serial_num=None,
        mac_addr=None,
        other=None,
        ip_addr=None,
        ch_names=None
    ):
        """The initialization function takes the name of the EEG device and determines whether or not
        the device belongs to the Muse or Brainflow families and initializes the appropriate backend.

        Parameters:
            device (str): name of eeg device used for reading data.

            ch_names (array_like or None): array containing custom specified channel names. Useful for custom montagues 
        like when external electrodes are used.
        """
        # determine if board uses brainflow or muselsl backend
        self.device_name = device
        self.serial_num = serial_num
        self.serial_port = serial_port
        self.mac_address = mac_addr
        self.ip_addr = ip_addr
        self.other = other
        self.backend = self._get_backend(self.device_name)
        self.initialize_backend()
        # Populate static device metadata when available; for generic LSL, metadata
        # is populated dynamically by the inlet and may not exist here.
        try:
            self.n_channels = len(EEG_INDICES[self.device_name])
            self.sfreq = SAMPLE_FREQS[self.device_name]
            self.channels = EEG_CHANNELS[self.device_name]
        except KeyError:
            pass
        self.ch_names = ch_names

    def initialize_backend(self):
        if self.backend == "brainflow":
            self._init_brainflow()
            self.timestamp_channel = BoardShim.get_timestamp_channel(
                self.brainflow_id)
        elif self.backend == "muselsl":
            self._init_muselsl()
            self._muse_get_recent()  # run this at initialization to get some
            # stream metadata into the eeg class
        elif self.backend == "lsl":
            # Generic LSL inlet for any EEG stream
            self._init_lsl()
            self._lsl_get_recent()  # prime stream metadata

    def _get_backend(self, device_name):
        if device_name in brainflow_devices:
            return "brainflow"
        elif device_name in ["muse2016", "muse2", "museS"]:
            return "muselsl"
        elif device_name in ["lsl", "LSL"]:
            return "lsl"

    #####################
    #   MUSE functions  #
    #####################
    def _init_muselsl(self):
        # Currently there's nothing we need to do here. However keeping the
        # option open to add things with this init method.
        self._muse_recent_inlet = None

    def _start_muse(self, duration):
        if sys.platform in ["linux", "linux2", "darwin"]:
            # Look for muses
            self.muses = list_muses()
            # self.muse = muses[0]

            # Start streaming process
            self.stream_process = Process(
                target=stream, args=(self.muses[0]["address"],)
            )
            self.stream_process.start()

        # Create markers stream outlet
        self.muse_StreamInfo = StreamInfo(
            "Markers", "Markers", 1, 0, "int32", "myuidw43536"
        )
        self.muse_StreamOutlet = StreamOutlet(self.muse_StreamInfo)

        # Start a background process that will stream data from the first available Muse
        print("starting background recording process")
        if self.save_fn:
            print("will save to file: %s" % self.save_fn)
        self.recording = Process(target=record, args=(duration, self.save_fn))
        self.recording.start()

        time.sleep(5)
        self.stream_started = True
        self.push_sample([99], timestamp=time.time())

    def _stop_muse(self):
        pass

    def _muse_push_sample(self, marker, timestamp):
        self.muse_StreamOutlet.push_sample(marker, timestamp)

    def _muse_get_recent(self, n_samples: int = 256, restart_inlet: bool = False):
        if self._muse_recent_inlet and not restart_inlet:
            inlet = self._muse_recent_inlet
        else:
            # Initiate a new lsl stream
            streams = resolve_byprop(
                "type", "EEG", timeout=mlsl_cnsts.LSL_SCAN_TIMEOUT)
            if not streams:
                raise Exception(
                    "Couldn't find any stream, is your device connected?")
            inlet = StreamInlet(
                streams[0], max_chunklen=mlsl_cnsts.LSL_EEG_CHUNK)
            self._muse_recent_inlet = inlet

        info = inlet.info()
        sfreq = info.nominal_srate()
        description = info.desc()
        n_chans = info.channel_count()

        self.sfreq = sfreq
        self.info = info
        self.n_chans = n_chans
        self.n_channels = n_chans

        timeout = (n_samples / sfreq) + 0.5
        samples, timestamps = inlet.pull_chunk(
            timeout=timeout, max_samples=n_samples)

        samples = np.array(samples)
        timestamps = np.array(timestamps)

        # Get EEG-only indices and labels if available
        eeg_indices, ch_names = _lsl_eeg_indices_and_labels(info)
        # Restrict to EEG channels only
        try:
            samples = samples[:, eeg_indices]
        except Exception:
            pass
        df = pd.DataFrame(samples, index=timestamps, columns=ch_names)
        return df

    #####################
    #   LSL functions   #
    #####################
    def _init_lsl(self):
        # Generic LSL: keep a reusable inlet for recent data pulls
        self._lsl_recent_inlet = None

    def _start_lsl(self, duration):
        # For generic LSL EEG streams we do not start the device stream here;
        # we only create a markers outlet and optionally record the incoming
        # EEG stream to file using the local LSL recorder utility.
        # Create markers stream outlet
        print("DEBUG: Creating LSL markers stream outlet...")
        self.lsl_StreamInfo = StreamInfo(
            "Markers", "Markers", 1, 0, "int32", "eegnb_markers")
        self.lsl_StreamOutlet = StreamOutlet(self.lsl_StreamInfo)
        print("DEBUG: LSL markers stream outlet created")

        # Give the marker stream time to become available before starting recorder
        time.sleep(1)

        # Start a lightweight background recording process from LSL to CSV if requested
        if self.save_fn:
            print(f"DEBUG: Starting LSL recording to {self.save_fn}")
            self._lsl_stop_event = Event()
            self.recording = Process(target=lsl_record_worker, args=(
                self.save_fn, self._lsl_stop_event))
            self.recording.start()
            print("DEBUG: LSL recording process started")

        # Allow stream buffers to fill a bit, then mark start
        time.sleep(2)
        self.stream_started = True
        # Let LSL stamp timestamps for proper alignment
        print("DEBUG: Pushing initial marker (99)")
        self.push_sample([99], timestamp=None)

    def _lsl_push_sample(self, marker, timestamp):
        # Push to the generic LSL marker outlet; ensure list-like
        if isinstance(marker, (int, np.integer)):
            marker = [int(marker)]
        print(f"DEBUG: Pushing marker {marker}")
        # Ignore provided timestamp; let LSL stamp using local clock for alignment
        try:
            self.lsl_StreamOutlet.push_sample(marker)
            print(f"DEBUG: Successfully pushed marker {marker}")
        except TypeError:
            from pylsl import local_clock
            self.lsl_StreamOutlet.push_sample(marker, local_clock())
            print(f"DEBUG: Successfully pushed marker {marker} with timestamp")

    def _lsl_get_recent(self, n_samples: int = 256, restart_inlet: bool = False):
        # Reuse inlet if available
        if self._lsl_recent_inlet and not restart_inlet:
            inlet = self._lsl_recent_inlet
        else:
            streams = resolve_byprop(
                "type", "EEG", timeout=mlsl_cnsts.LSL_SCAN_TIMEOUT)
            if not streams:
                raise Exception(
                    "Couldn't find any LSL EEG stream. Is your device streaming?")
            inlet = StreamInlet(
                streams[0], max_chunklen=mlsl_cnsts.LSL_EEG_CHUNK)
            self._lsl_recent_inlet = inlet

        info = inlet.info()
        sfreq = info.nominal_srate()
        description = info.desc()
        n_chans = info.channel_count()

        self.sfreq = sfreq
        self.info = info
        self.n_chans = n_chans
        self.n_channels = n_chans

        timeout = (n_samples / sfreq) + 0.5 if sfreq else 2.0
        samples, timestamps = inlet.pull_chunk(
            timeout=timeout, max_samples=n_samples)

        samples = np.array(samples)
        timestamps = np.array(timestamps)

        # Attempt to parse channel labels from stream metadata; fallback to generic names
        ch_names = []
        try:
            ch = description.child("channels").first_child()
            if ch:  # if metadata present
                ch_names = [ch.child_value("label")]
                for i in range(n_chans - 1):
                    ch = ch.next_sibling()
                    lab = ch.child_value("label")
                    if lab != "":
                        ch_names.append(lab)
        except Exception:
            ch_names = []

        if not ch_names or len(ch_names) != n_chans:
            ch_names = [f"eeg_{i}" for i in range(n_chans)]

        df = pd.DataFrame(samples, index=timestamps, columns=ch_names)
        return df

    def _stop_lsl(self):
        """Gracefully stop LSL-related resources.

        Wait for the background muselsl recorder (if any) to finish before
        tearing down the marker outlet to avoid reconnect spam on shutdown.
        """
        # Best-effort: send a trailing marker to denote end of run
        try:
            self.push_sample([0], timestamp=None)
        except Exception:
            pass

        # If a background recording process was started, signal stop and wait briefly for it to finish
        rec = getattr(self, "recording", None)
        if isinstance(rec, Process):
            try:
                # Signal the recorder to finish and flush
                stop_evt = getattr(self, "_lsl_stop_event", None)
                if stop_evt is not None:
                    stop_evt.set()
                if rec.is_alive():
                    logger.info("Waiting for LSL recording process to finish…")
                    rec.join(timeout=10)
                if rec.is_alive():
                    logger.warning("LSL recording still running; terminating.")
                    rec.terminate()
                    rec.join(timeout=3)
            finally:
                self.recording = None
                if hasattr(self, "_lsl_stop_event"):
                    self._lsl_stop_event = None

        # Tear down marker outlet/info to release resources
        try:
            if hasattr(self, "lsl_StreamOutlet"):
                self.lsl_StreamOutlet = None
            if hasattr(self, "lsl_StreamInfo"):
                self.lsl_StreamInfo = None
        except Exception:
            pass

        self.stream_started = False

    ##########################
    #   BrainFlow functions  #
    ##########################
    def _init_brainflow(self):
        """This function initializes the brainflow backend based on the input device name. It calls
        a utility function to determine the appropriate USB port to use based on the current operating system.
        Additionally, the system allows for passing a serial number in the case that they want to use either
        the BraintBit or the Unicorn EEG devices from the brainflow family.

        Parameters:
             serial_num (str or int): serial number for either the BrainBit or Unicorn devices.
        """
        # Initialize brainflow parameters
        self.brainflow_params = BrainFlowInputParams()

        if self.device_name == "ganglion":
            self.brainflow_id = BoardIds.GANGLION_BOARD.value
            if self.serial_port is None:
                self.brainflow_params.serial_port = get_openbci_usb()
            # set mac address parameter in case
            if self.mac_address is None:
                print("No MAC address provided, attempting to connect without one")
            else:
                self.brainflow_params.mac_address = self.mac_address

        elif self.device_name == "ganglion_wifi":
            self.brainflow_id = BoardIds.GANGLION_WIFI_BOARD.value
            if self.ip_addr is not None:
                self.brainflow_params.ip_address = self.ip_addr
                self.brainflow_params.ip_port = 6677

        elif self.device_name == "cyton":
            self.brainflow_id = BoardIds.CYTON_BOARD.value
            if self.serial_port is None:
                self.brainflow_params.serial_port = get_openbci_usb()

        elif self.device_name == "cyton_wifi":
            self.brainflow_id = BoardIds.CYTON_WIFI_BOARD.value
            if self.ip_addr is not None:
                self.brainflow_params.ip_address = self.ip_addr
                self.brainflow_params.ip_port = 6677

        elif self.device_name == "cyton_daisy":
            self.brainflow_id = BoardIds.CYTON_DAISY_BOARD.value
            if self.serial_port is None:
                self.brainflow_params.serial_port = get_openbci_usb()

        elif self.device_name == "cyton_daisy_wifi":
            self.brainflow_id = BoardIds.CYTON_DAISY_WIFI_BOARD.value
            if self.ip_addr is not None:
                self.brainflow_params.ip_address = self.ip_addr

        elif self.device_name == "brainbit":
            self.brainflow_id = BoardIds.BRAINBIT_BOARD.value

        elif self.device_name == "unicorn":
            self.brainflow_id = BoardIds.UNICORN_BOARD.value

        elif self.device_name == "callibri_eeg":
            self.brainflow_id = BoardIds.CALLIBRI_EEG_BOARD.value
            if self.other:
                self.brainflow_params.other_info = str(self.other)

        elif self.device_name == "notion1":
            self.brainflow_id = BoardIds.NOTION_1_BOARD.value

        elif self.device_name == "notion2":
            self.brainflow_id = BoardIds.NOTION_2_BOARD.value

        elif self.device_name == "crown":
            self.brainflow_id = BoardIds.CROWN_BOARD.value

        elif self.device_name == "freeeeg32":
            self.brainflow_id = BoardIds.FREEEEG32_BOARD.value
            if self.serial_port is None:
                self.brainflow_params.serial_port = get_openbci_usb()

        elif self.device_name == "museS_bfn":
            self.brainflow_id = BoardIds.MUSE_S_BOARD.value

        elif self.device_name == "museS_bfb":
            self.brainflow_id = BoardIds.MUSE_S_BLED_BOARD.value

        elif self.device_name == "muse2_bfn":
            self.brainflow_id = BoardIds.MUSE_2_BOARD.value

        elif self.device_name == "muse2_bfb":
            self.brainflow_id = BoardIds.MUSE_2_BLED_BOARD.value

        elif self.device_name == "muse2016_bfn":
            self.brainflow_id = BoardIds.MUSE_2016_BOARD.value

        elif self.device_name == "muse2016_bfb":
            self.brainflow_id = BoardIds.MUSE_2016_BLED_BOARD.value

        elif self.device_name == "synthetic":
            self.brainflow_id = BoardIds.SYNTHETIC_BOARD.value

        # some devices allow for an optional serial number parameter for better connection
        if self.serial_num:
            serial_num = str(self.serial_num)
            self.brainflow_params.serial_number = serial_num

        if self.serial_port:
            serial_port = str(self.serial_port)
            self.brainflow_params.serial_port = serial_port

        # Initialize board_shim
        self.sfreq = BoardShim.get_sampling_rate(self.brainflow_id)
        self.board = BoardShim(self.brainflow_id, self.brainflow_params)
        self.board.prepare_session()

    def _start_brainflow(self):
        # only start stream if non exists
        if not self.stream_started:
            self.board.start_stream()

        self.stream_started = True

        # wait for signal to settle
        if (self.device_name.find("cyton") != -1) or (
            self.device_name.find("ganglion") != -1
        ):
            # wait longer for openbci cyton / ganglion
            sleep(10)
        else:
            sleep(5)

    def _stop_brainflow(self):
        """This functions kills the brainflow backend and saves the data to a CSV file."""

        # Collect session data and kill session
        data = self.board.get_board_data()  # will clear board buffer
        self.board.stop_stream()
        self.board.release_session()

        # Extract relevant metadata from board
        ch_names, eeg_data, timestamps = self._brainflow_extract(data)

        # Create a column for the stimuli to append to the EEG data
        stim_array = create_stim_array(timestamps, self.markers)
        timestamps = timestamps[..., None]

        # Add an additional dimension so that shapes match
        total_data = np.append(timestamps, eeg_data, 1)

        # Append the stim array to data.
        total_data = np.append(total_data, stim_array, 1)

        # Subtract five seconds of settling time from beginning
        total_data = total_data[5 * self.sfreq:]
        data_df = pd.DataFrame(total_data, columns=[
                               "timestamps"] + ch_names + ["stim"])
        data_df.to_csv(self.save_fn, index=False)

    def _brainflow_extract(self, data):
        """
        Formats the data returned from brainflow to get
        ch_names; list of channel names
        eeg_data: NDArray of eeg samples
        timestamps: NDArray of timestamps
        """

        # transform data for saving
        data = data.T  # transpose data

        # explicitly assign channel names for EEG data
        if self.ch_names is not None:
            ch_names = self.ch_names
        # automatically assign the channel names for EEG data
        elif (
            self.brainflow_id == BoardIds.GANGLION_BOARD.value
            or self.brainflow_id == BoardIds.GANGLION_WIFI_BOARD.value
        ):
            # if a ganglion is used, use recommended default EEG channel names
            ch_names = ["fp1", "fp2", "tp7", "tp8"]
        elif self.brainflow_id == BoardIds.FREEEEG32_BOARD.value:
            ch_names = [f"eeg_{i}" for i in range(0, 32)]
        else:
            # otherwise select eeg channel names via brainflow API
            ch_names = BoardShim.get_eeg_names(self.brainflow_id)

        # pull EEG channel data via brainflow API
        eeg_data = data[:, BoardShim.get_eeg_channels(self.brainflow_id)]
        timestamps = data[:, BoardShim.get_timestamp_channel(
            self.brainflow_id)]

        return ch_names, eeg_data, timestamps

    def _brainflow_push_sample(self, marker):
        last_timestamp = self.board.get_current_board_data(
            1)[self.timestamp_channel][0]
        self.markers.append([marker, last_timestamp])

    def _brainflow_get_recent(self, n_samples=256):

        # initialize brainflow if not set
        if self.board == None:
            self._init_brainflow()

        # start branflow stream
        self._start_brainflow()

        # get the latest data
        data = self.board.get_current_board_data(n_samples)

        ch_names, eeg_data, timestamps = self._brainflow_extract(data)

        eeg_data = np.array(eeg_data)
        timestamps = np.array(timestamps)

        df = pd.DataFrame(eeg_data, index=timestamps, columns=ch_names)
        # print (df)
        return df

    #################################
    #   Highlevel device functions  #
    #################################

    def start(self, fn, duration=None):
        """Starts the EEG device based on the defined backend.

        Parameters:
            fn (str): name of the file to save the sessions data to.
        """
        if fn:
            self.save_fn = fn

        if self.backend == "brainflow":
            self._start_brainflow()
            self.markers = []
        elif self.backend == "muselsl":
            self._start_muse(duration)
        elif self.backend == "lsl":
            self._start_lsl(duration)

    def push_sample(self, marker, timestamp):
        """
        Universal method for pushing a marker and its timestamp to store alongside the EEG data.

        Parameters:
            marker (int): marker number for the stimuli being presented.
            timestamp (float): timestamp of stimulus onset from time.time() function.
        """
        if self.backend == "brainflow":
            self._brainflow_push_sample(marker=marker)
        elif self.backend == "muselsl":
            self._muse_push_sample(marker=marker, timestamp=timestamp)
        elif self.backend == "lsl":
            self._lsl_push_sample(marker=marker, timestamp=timestamp)

    def stop(self):
        if self.backend == "brainflow":
            self._stop_brainflow()
        elif self.backend == "muselsl":
            pass
        elif self.backend == "lsl":
            self._stop_lsl()

    def get_recent(self, n_samples: int = 256):
        """
        Usage:
        -------
        from eegnb.devices.eeg import EEG
        this_eeg = EEG(device='museS')
        df_rec = this_eeg.get_recent()
        """

        if self.backend == "brainflow":
            df = self._brainflow_get_recent(n_samples)
        elif self.backend == "muselsl":
            df = self._muse_get_recent(n_samples)
        elif self.backend == "lsl":
            df = self._lsl_get_recent(n_samples)
        else:
            raise ValueError(f"Unknown backend {self.backend}")

        # Sort out the sensor coils
        sorted_cols = sorted(df.columns)
        df = df[sorted_cols]

        return df
