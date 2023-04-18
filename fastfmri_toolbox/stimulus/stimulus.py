from typing import Union, List, Tuple

import numpy as np


class Stimulus:
    def __init__(
        self,
        TR: float,
        n_timepoints: int,
        stimulus_start_time: float,
        stimulus_duration: float,
        stimulus_frequencies: Union[float, List[float]],
        stimulus_amplitude: float = 1.0,
        stimulus_start_on: bool = False,
    ):
        self.TR = TR
        self.n_timepoints = n_timepoints
        self.start_time = stimulus_start_time
        self.duration = stimulus_duration
        self.end_time = self.start_time + self.duration
        self.frequencies = self._read_stimulus_frequencies(stimulus_frequencies)
        self.amplitude = stimulus_amplitude
        self.start_on = stimulus_start_on

    def generate(
        self,
        normalize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Note: stimulus start time may not coincide a TR.
        As such, a phase shift time offset is added to the
        sine wave.
        """
        n_timepoints_stim_off = int(self.start_time / self.TR)
        delay = self.start_time - self.TR * n_timepoints_stim_off  # seconds
        DELAY_ERROR_MSG = (
            "Delay (seconds) between acquired fMRI volume and stimulus start time "
            "must be positive."
        )
        assert delay > 0, DELAY_ERROR_MSG

        # Stimulus OFF (@ start)
        if self.start_on:
            stim_off_start = 2 * self.amplitude * np.ones((n_timepoints_stim_off,))
        else:
            stim_off_start = np.zeros((n_timepoints_stim_off,))
        # Stimulus ON
        for ix, f in enumerate(self.frequencies):
            if ix == 0:
                stim_on = self._sample_periodic_signal(f, delay, self.duration, self.TR)
            else:
                stim_on += self._sample_periodic_signal(
                    f, delay, self.duration, self.TR
                )
        if normalize:
            stim_on = (stim_on - stim_on.min()) / (stim_on.max() - stim_on.min())
        # Stimulus OFF (@ end)
        stim_off_end = np.zeros(
            (self.n_timepoints - stim_off_start.shape[0] - stim_on.shape[0],)
        )

        # combine stimulus on/off
        self.stim_signal = np.concatenate((stim_off_start, stim_on, stim_off_end))

        # Stimulus timepoint grid
        self.stim_time = np.arange(0, self.TR * self.n_timepoints, self.TR)

        return (self.stim_time, self.stim_signal)

    def convolve(self, hrf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not hasattr(self, "stim_signal"):
            self.generate()

        convolved_signal = np.convolve(self.stim_signal, hrf)
        # Zero pad stimulus arrays to account for convolution kernel
        N_diff = convolved_signal.shape[0] - self.stim_signal.shape[0]
        updated_stim = np.concatenate((self.stim_signal, np.zeros((N_diff))))
        updated_stim_time = np.round(
            np.arange(0, self.TR * (self.n_timepoints + N_diff), self.TR),
            self._round_TR(self.TR),
        )

        return (updated_stim_time, updated_stim, convolved_signal)

    def _read_stimulus_frequencies(
        self, stim_frequencies: Union[float, List[float]]
    ) -> List[float]:
        if isinstance(stim_frequencies, float):
            return [stim_frequencies]

        else:
            return stim_frequencies

    def _sample_periodic_signal(
        self,
        f: float,
        delay: float,
        stimulus_duration: float,
        TR: float,
    ) -> np.ndarray:
        stim_time_on = np.arange(0, stimulus_duration, TR)
        stim_tr_offset = 2 * f * np.pi * delay
        stim_offset = np.pi / 2  # Periodic stimulus begins at 0
        stim_on = (
            self.amplitude
            * np.sin(2 * np.pi * f * stim_time_on - stim_offset - stim_tr_offset)
            + self.amplitude
        )
        stim_on[0] = 0  # Stimulus is still off @ first volume

        return stim_on

    def _round_TR(self, TR: Union[int, float]) -> int:
        if isinstance(TR, int):
            TR = float(TR)

        return len(str(TR).split(".")[1])
