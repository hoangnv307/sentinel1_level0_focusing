"""Kiểm chứng RRF và thử hai phổ mở rộng giả thuyết; không nhận là công thức IPF."""

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft, fftshift
import sentinel1decoder

import estimate_doppler_centroid as demo
from audit_doppler_centroid import compare, fingerprint


def candidate_reference(reference, sample_rate_hz, pulse_length_s, fft_length, mode):
    """Giả thuyết stationary-phase LFM, không phải triển khai IPF đã xác nhận."""
    phase = reference.phase
    rate = 2 * phase[2]
    bandwidth = abs(rate) * pulse_length_s
    if (mode not in ("flat", "cosine") or rate == 0 or phase[3] != 0
            or not np.array_equal(reference.amplitude, [1, 0, 0, 0])
            or bandwidth + 2*abs(phase[1]) >= sample_rate_hz):
        raise ValueError("Candidate chỉ hỗ trợ nominal LFM phẳng nằm trong Nyquist.")
    f = fftfreq(fft_length, d=1/sample_rate_hz)
    n = int(np.ceil(sample_rate_hz*pulse_length_s))
    centre_time = (n-1)/(2*sample_rate_hz)
    angle = (np.pi*(f-phase[1])**2/rate - 2*np.pi*f*centre_time
             - 2*np.pi*phase[0] - np.sign(rate)*np.pi/4
             + 2*np.pi*f*reference.internal_delay_s)
    h = np.exp(1j*angle)
    if mode == "cosine":
        low, high = phase[1]-bandwidth/2, phase[1]+bandwidth/2
        distance = np.maximum((low-f)/(low+sample_rate_hz/2),
                              (f-high)/(sample_rate_hz/2-high))
        h *= .5*(1+np.cos(np.pi*np.clip(distance, 0, 1)))
    h /= np.sqrt(np.mean(abs(h)**2))
    return (h*reference.replica_gain).astype(np.complex64)


def spectrum_checks(reference, rate, pulse_length, raw_samples):
    n = int(np.ceil(rate*pulse_length))
    length = 1 << (raw_samples+n-2).bit_length()
    f = fftfreq(length, 1/rate)
    native = reference.range_reference(rate, pulse_length, length)
    bank = {"aux_fft": native, **{
        mode: candidate_reference(reference, rate, pulse_length, length, mode)
        for mode in ("flat", "cosine")}}
    rng = np.random.default_rng(20261021)
    raw = rng.normal(size=(4, raw_samples)) + 1j*rng.normal(size=(4, raw_samples))
    figures, axes = plt.subplots(2, 1, figsize=(11, 7))
    results = {}
    for name, h in bank.items():
        factory = reference.range_reference if name == "aux_fft" else (
            lambda fs, tx, size, mode=name: candidate_reference(reference, fs, tx, size, mode))
        h2 = factory(rate, pulse_length, 2*length)
        y1 = ifft(fft(raw, n=length, axis=1)*h, axis=1)[:, n-1:raw_samples-1]
        y2 = ifft(fft(raw, n=2*length, axis=1)*h2, axis=1)[:, n-1:raw_samples-1]
        # Cùng vùng bên trong cho tất cả candidate; đây là guard nghiên cứu, không là IPF setting.
        inner = slice(512, y1.shape[1]-512)
        delta = y1[:, inner]-y2[:, inner]
        bandwidth = 2*abs(reference.phase[2])*pulse_length
        outside = abs(f-reference.phase[1]) > bandwidth/2
        results[name] = {
            "energy_after_gain": float(np.mean(abs(h.astype(complex))**2)),
            "time_energy_after_gain": float(np.sum(abs(ifft(h).astype(complex))**2)),
            "energy_outside_nominal_band_fraction": float(np.sum(abs(h[outside])**2)/np.sum(abs(h)**2)),
            "fft_doubling_relative_rms_inner": float(np.sqrt(np.mean(abs(delta)**2)/np.mean(abs(y2[:, inner])**2))),
            "fft_doubling_max_abs_inner": float(np.max(abs(delta))),
        }
        axes[0].plot(fftshift(f)/1e6, fftshift(20*np.log10(np.maximum(abs(h), 1e-12))), label=name)
        impulse = fftshift(abs(ifft(h)))
        axes[1].plot(np.arange(-length//2, length//2), impulse, label=name)
    axes[0].set(xlabel="Range frequency [MHz]", ylabel="RRF magnitude [dB]")
    axes[1].set(xlabel="Kernel sample index (FFT shifted)", ylabel="Impulse magnitude", xlim=(-400, n+400))
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend()
    figures.tight_layout()
    return {"fft_length": length, "pulse_samples": n, "interior_guard_samples": 512,
            "spectra": results}, figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, default=Path.home()/"sar_data/focusing_data/product_1")
    parser.add_argument("--output", type=Path, default=Path("output/sentinel-1/rrf_check"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = demo.one(args.product.glob("level-0/s1a-s4-raw-s-vv-*.dat"), "raw")
    ann_path = demo.one(args.product.glob("level-1/s1a-s4-slc-vv-*.xml"), "annotation")
    annotation = ET.parse(ann_path).getroot()
    aux = demo.read_auxiliary(args.product)
    reference = demo.ReferenceInput(annotation, aux)
    level0 = sentinel1decoder.Level0File(str(raw))
    header = level0.get_acquisition_chunk_metadata(13).iloc[0]
    rate = sentinel1decoder.utilities.range_dec_to_sample_rate(header["Range Decimation"])
    pulse_length = float(header["Tx Pulse Length"])
    checks, figure = spectrum_checks(reference, rate, pulse_length, 2*int(header["Number of Quads"]))
    figure.savefig(args.output/"spectra.png", dpi=160)
    plt.close(figure)
    report = {"reference_input": reference.summary(), "checks": checks, "cases": [],
              "candidate_status": "Research hypotheses, not confirmed IPF Extended Tapered",
              "fingerprints": [fingerprint(p) for p in (raw, ann_path, *map(Path, aux["files"]),
                  Path(__file__), Path(demo.__file__),
                  Path("src/sentinel1_processing/range_processing/range_reference_function.py"),
                  Path("src/sentinel1_processing/utils/dce_reference_input.py"))]}
    geometry = demo.geometry.Estimator.from_level0_product(level0, 299792458/aux["radar_frequency_hz"])
    annotated = annotation.findall("dopplerCentroid/dcEstimateList/dcEstimate")
    cases = [("annotation_fft", copy.deepcopy(reference)), ("aux_fft", reference)]
    cases[0][1].amplitude = reference.annotated_amplitude
    cases[0][1].phase = reference.annotated_phase
    for mode in ("flat", "cosine"):
        candidate = copy.deepcopy(reference)
        candidate.range_reference = lambda fs, tx, size, mode=mode: candidate_reference(reference, fs, tx, size, mode)
        cases.append((mode, candidate))
    for name, correction in cases:
        with TemporaryDirectory(prefix="rrf-", dir=args.output) as directory:
            segments, pri, _, diagnostics = demo.compress_echoes(level0, aux, directory, correction)
            config, starts, t0, _ = demo.infer_configuration(
                annotation, segments, pri, rate, aux["rms_threshold_hz"],
                diagnostics[0]["hardware_delay_far_throwaway_samples"])
            guards = (0, 512) if name in ("annotation_fft", "aux_fft") else (512,)
            for guard in guards:
                used = replace(config, range_roi_start=guard, range_roi_stop=config.range_roi_stop-guard)
                estimates = demo.estimation.Estimator(1/pri, used).estimate_segments(
                    segments, custom_azimuth_starts=starts, dce_range_start_s=t0, t0_s=t0,
                    geometry_dc_provider=lambda eta, tau: geometry.estimate(eta, tau*299792458/2))
                blocks = [compare(r, a) for r, a in zip(estimates, annotated)]
                report["cases"].append({"name": name, "extra_guard_samples": guard,
                                        "config": asdict(used), "input_diagnostics": diagnostics,
                                        "blocks": blocks})
                (args.output/"rrf_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
                print(name, guard, [(b["fine_rmse_hz"], b["fit_rms_hz"]) for b in blocks], flush=True)
            del segments


if __name__ == "__main__":
    main()
