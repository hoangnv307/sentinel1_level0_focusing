"""RRF: năng lượng, whitening, gain/delay và gốc range theo DAD §6.1.1."""

import unittest

import numpy as np
from scipy.fft import fft, fftfreq, ifft

from sentinel1_processing.range_processing.range_reference_function import from_coefficients, create_freq_domain
from sentinel1_processing.azimuth_pre_processing.range.compression import compress


class RangeReferenceTest(unittest.TestCase):
    def make(self, **overrides):
        config = dict(sample_rate_hz=128., pulse_length_s=0.5, fft_length=512,
                      amplitude_coefficients=[2., 1.], phase_coefficients_cycles=[0.3, 0., 96.])
        return from_coefficients(**{**config, **overrides})

    def test_parseval_energy_and_fft_length_independence(self):
        for length in (256, 512, 1024):
            h = self.make(fft_length=length)
            self.assertAlmostEqual(float(np.mean(abs(h.astype(complex))**2)), 1., places=6)
            self.assertAlmostEqual(float(np.sum(abs(ifft(h).astype(complex))**2)), 1., places=6)

    def test_whitening_is_inverse_amplitude_before_energy_normalization(self):
        t = (np.arange(64) - 31.5) / 128
        expected = fft((np.exp(-2j * np.pi * (0.3 + 96*t**2)) / (2+t))[::-1], n=512)
        expected /= np.sqrt(np.mean(abs(expected)**2))
        np.testing.assert_allclose(self.make(), expected, rtol=2e-6, atol=2e-7)

    def test_delay_sign_and_complex_gain_after_normalization(self):
        delay = 7.25 / 128
        gain = 2 + 3j
        h = self.make(internal_delay_s=delay, complex_gain=gain)
        expected = self.make() * np.exp(2j*np.pi*fftfreq(512, 1/128)*delay) * gain
        np.testing.assert_allclose(h, expected, rtol=3e-6, atol=5e-7)
        self.assertAlmostEqual(float(np.mean(abs(h.astype(complex))**2)), abs(gain)**2, places=5)

    def test_odd_even_replica_keep_the_physical_range_origin(self):
        for n in (63, 64):
            fs = 128.
            phase = [0., 1., 96.]
            t = (np.arange(n) - (n-1)/2) / fs
            chirp = np.exp(2j*np.pi*np.polynomial.polynomial.polyval(t, phase))
            raw = np.zeros((1, 256), dtype=np.complex64)
            raw[0, 30:30+n] = chirp
            tau = .005 + np.arange(256)/fs
            h = from_coefficients(sample_rate_hz=fs, pulse_length_s=n/fs, fft_length=512,
                                  amplitude_coefficients=[1.], phase_coefficients_cycles=phase,
                                  internal_delay_s=7/fs)
            data, axis = compress(raw, tau, sample_rate_hz=fs, pulse_length_s=n/fs,
                                 pulse_start_frequency_hz=0., pulse_ramp_rate_hz_per_s=0.,
                                 range_reference_function=h, output="slc")
            self.assertEqual(int(np.argmax(abs(data[0]))), 23)
            np.testing.assert_array_equal(axis, tau[:256-n])

    def test_rejects_invalid_replica_parameters(self):
        for change in ({"sample_rate_hz": 0.}, {"pulse_length_s": np.nan},
                       {"fft_length": 32}, {"internal_delay_s": -1.},
                       {"amplitude_coefficients": [0.]}, {"phase_coefficients_cycles": [np.inf]},
                       {"complex_gain": 0j}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.make(**change)

    def test_header_rrf_removes_known_integer_and_fractional_forward_delay(self):
        fs, n, size, length = 128., 64, 256, 512
        f0, ramp = -24., 96.
        time = (np.arange(n)-(n-1)/2)/fs
        chirp = np.exp(2j*np.pi*((f0+ramp*n/(2*fs))*time+.5*ramp*time**2))
        transmitted = np.zeros(length, complex)
        transmitted[60:60+n] = chirp
        expected = np.convolve(transmitted[:size], chirp[::-1].conj()/np.sqrt(n))[n-1:size-1]
        for delay_samples in (7., 7.25, 7.8):
            # Independent forward instrument: negative ramp delays the acquired chirp.
            raw = ifft(fft(transmitted)*np.exp(-2j*np.pi*fftfreq(length)*delay_samples))[:size]
            reference = create_freq_domain(sample_rate_hz=fs, pulse_length_s=n/fs,
                pulse_start_frequency_hz=f0, pulse_ramp_rate_hz_per_s=ramp,
                fft_length=length, internal_delay_s=delay_samples/fs)
            result, axis = compress(raw[None, :], np.arange(size)/fs, sample_rate_hz=fs,
                pulse_length_s=n/fs, pulse_start_frequency_hz=f0, pulse_ramp_rate_hz_per_s=ramp,
                range_reference_function=reference, output="slc")
            self.assertEqual(int(np.argmax(abs(result[0]))), 60)
            # Finite acquisition truncates fractional-shift sinc tails outside the window.
            np.testing.assert_allclose(result[0, 20:150], expected[20:150], atol=2e-5, rtol=0)
            np.testing.assert_array_equal(axis, np.arange(size-n)/fs)


if __name__ == "__main__":
    unittest.main()
