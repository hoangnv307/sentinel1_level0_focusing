import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Focus và ghép SLC cho chunk 13–14

    Notebook này range-compress hai chunk, đưa chunk 14 về lưới slant-range của
    chunk 13, rồi focus toàn bộ dải azimuth một lần. Các mảng lớn được lưu dưới
    dạng `.npy`/memmap để không phải giữ nhiều bản sao trong RAM.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    from time import perf_counter

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np
    import sentinel1decoder

    from notebook_support.cache import (
        array_cache_matches,
        cache_fingerprint,
        chunk_cache_key,
        open_array,
        save_cache_fingerprint,
    )

    notebook_started_at = perf_counter()
    plt.style.use("default")
    return (
        PROJECT_ROOT,
        Path,
        colors,
        array_cache_matches,
        cache_fingerprint,
        chunk_cache_key,
        notebook_started_at,
        np,
        open_array,
        perf_counter,
        plt,
        sentinel1decoder,
        save_cache_fingerprint,
    )


@app.cell
def _(Path):
    import sentinel1_processing.azimuth_pre_processing as azimuth_pre_processing
    import sentinel1_processing.azimuth_processing as azimuth_processing
    import sentinel1_processing.doppler_centroid_estimation as doppler_centroid_estimation
    import sentinel1_processing.effective_velocity as effective_velocity
    import sentinel1_processing.range_processing as range_processing
    import sentinel1_processing.raw_data_correction as raw_data_correction
    import sentinel1_processing.s6_parameters as s6_parameters

    _range_roots = (
        Path(azimuth_pre_processing.__file__).parent,
        Path(range_processing.__file__).parent,
    )
    range_source = tuple(
        _path.read_bytes()
        for _root in dict.fromkeys(_range_roots)
        for _path in sorted(_root.rglob("*.py"))
    ) + (Path(s6_parameters.__file__).read_bytes(),)
    raw_correction_source = Path(raw_data_correction.__file__).read_bytes()
    doppler_source = (
        Path(doppler_centroid_estimation.__file__).read_bytes(),
        Path(s6_parameters.__file__).read_bytes(),
    )
    _focus_roots = (Path(azimuth_processing.__file__).parent,)
    focus_source = tuple(
        _path.read_bytes()
        for _root in _focus_roots
        for _path in sorted(_root.rglob("*.py"))
    ) + (
        Path(effective_velocity.__file__).read_bytes(),
        Path(s6_parameters.__file__).read_bytes(),
    )
    return (
        azimuth_pre_processing,
        azimuth_processing,
        doppler_source,
        doppler_centroid_estimation,
        effective_velocity,
        focus_source,
        range_processing,
        range_source,
        raw_correction_source,
        raw_data_correction,
        s6_parameters,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 - Dữ liệu vào và tham số radar
    """)
    return


@app.cell
def _(PROJECT_ROOT, mo, sentinel1decoder):
    CACHE_ROOT = str(PROJECT_ROOT / ".cache" / "sentinel1")
    _input_path = (
        PROJECT_ROOT
        / "data"
        / "sao_paulo"
        / "s1a-s6-raw-s-vv-20251226t214356-20251226t214427-062491-07d496.dat"
    )
    _watched_input = mo.watch.file(str(_input_path))
    _input_stat = _watched_input.stat()
    input_identity = (
        str(_watched_input.resolve()),
        _input_stat.st_size,
        _input_stat.st_mtime_ns,
    )
    l0file = sentinel1decoder.Level0File(str(_input_path))
    return CACHE_ROOT, input_identity, l0file


@app.cell
def _(chunk_cache_key, l0file, s6_parameters, sentinel1decoder):
    CHUNKS = (13, 14)
    CHUNK_CACHE_KEY = chunk_cache_key(CHUNKS)
    metadata_13 = l0file.get_acquisition_chunk_metadata(CHUNKS[0])
    metadata_14 = l0file.get_acquisition_chunk_metadata(CHUNKS[1])

    c = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS
    wavelength_m = s6_parameters.RADAR_WAVELENGTH_M
    PRI = float(metadata_13["PRI"].iloc[0])
    RGDEC = metadata_13["Range Decimation"].iloc[0]
    TXPSF = float(metadata_13["Tx Pulse Start Frequency"].iloc[0])
    TXPRR = float(metadata_13["Tx Ramp Rate"].iloc[0])
    TXPL = float(metadata_13["Tx Pulse Length"].iloc[0])
    range_sample_freq = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    range_sample_period = 1.0 / range_sample_freq
    az_sample_freq = 1.0 / PRI
    suppressed_data_time = 320.0 / (8.0 * sentinel1decoder.constants.F_REF)
    return (
        CHUNKS,
        CHUNK_CACHE_KEY,
        PRI,
        TXPL,
        TXPRR,
        TXPSF,
        az_sample_freq,
        c,
        metadata_13,
        metadata_14,
        range_sample_freq,
        range_sample_period,
        suppressed_data_time,
        wavelength_m,
    )


@app.cell
def _(
    PRI,
    metadata_13,
    metadata_14,
    np,
    range_processing,
    range_sample_freq,
    s6_parameters,
    suppressed_data_time,
):
    SWST_BIAS_S = s6_parameters.SWST_BIAS_S

    def _axes(metadata):
        _count = 2 * int(metadata["Number of Quads"].iloc[0])
        _raw_tau = (
            metadata["Rank"].iloc[0] * PRI
            + metadata["SWST"].iloc[0]
            + suppressed_data_time
            + np.arange(_count) / range_sample_freq
        )
        _tau = range_processing.swst_bias.correct(_raw_tau, SWST_BIAS_S)
        _eta = (
            metadata["Coarse Time"].to_numpy(dtype=float)
            + metadata["Fine Time"].to_numpy(dtype=float)
        )
        return _tau, _eta, _count

    raw_tau_13, eta_13, raw_range_count_13 = _axes(metadata_13)
    raw_tau_14, eta_14, raw_range_count_14 = _axes(metadata_14)
    _common_start = min(raw_tau_13[0], raw_tau_14[0])

    def _fractional_shift(tau):
        _offset = (tau[0] - _common_start) * range_sample_freq
        return (round(_offset) - _offset) / range_sample_freq

    range_time_shift_13 = _fractional_shift(raw_tau_13)
    range_time_shift_14 = _fractional_shift(raw_tau_14)
    return (
        SWST_BIAS_S,
        eta_13,
        eta_14,
        raw_range_count_13,
        raw_range_count_14,
        range_time_shift_13,
        range_time_shift_14,
        raw_tau_13,
        raw_tau_14,
    )


@app.cell
def _(
    TXPL,
    TXPRR,
    TXPSF,
    np,
    range_processing,
    range_sample_freq,
    raw_range_count_14,
):
    transmitted_pulse_samples = int(np.ceil(TXPL * range_sample_freq))
    _max_raw_samples = raw_range_count_14
    range_fft_length = 1 << int(np.ceil(np.log2(
        _max_raw_samples + transmitted_pulse_samples - 1
    )))
    range_reference_function = range_processing.reference_function.calculate(
        sample_rate_hz=range_sample_freq,
        pulse_start_frequency_hz=TXPSF,
        pulse_ramp_rate_hz_per_s=TXPRR,
        pulse_length_s=TXPL,
        fft_length=range_fft_length,
    )
    return range_reference_function, transmitted_pulse_samples


@app.cell
def _(
    Path,
    TXPL,
    TXPRR,
    TXPSF,
    azimuth_pre_processing,
    l0file,
    np,
    range_reference_function,
    range_sample_freq,
    raw_data_correction,
    transmitted_pulse_samples,
):
    def compress_chunk_to_file(chunk, raw_tau, range_time_shift_s, destination):
        _radar_data = l0file.get_acquisition_chunk_data(chunk)
        _iq_bias = np.complex128(
            raw_data_correction.estimate_iq_bias(_radar_data)
        )
        _path = Path(destination)
        _path.parent.mkdir(parents=True, exist_ok=True)
        _temporary = _path.with_suffix(f"{_path.suffix}.tmp")
        _shape = (
            _radar_data.shape[0],
            _radar_data.shape[1] - transmitted_pulse_samples + 1,
        )
        _output = np.lib.format.open_memmap(
            _temporary, mode="w+", dtype=np.complex64, shape=_shape
        )
        _compressed, _range_times = (
            azimuth_pre_processing.range.compression.compress(
                _radar_data,
                raw_tau,
                sample_rate_hz=range_sample_freq,
                pulse_start_frequency_hz=TXPSF,
                pulse_ramp_rate_hz_per_s=TXPRR,
                pulse_length_s=TXPL,
                iq_bias=_iq_bias,
                range_reference_function=range_reference_function,
                range_time_shift_s=range_time_shift_s,
                output_array=_output,
            )
        )
        _compressed.flush()
        del _compressed, _output
        _temporary.replace(_path)
        return _range_times, (float(_iq_bias.real), float(_iq_bias.imag))

    return (compress_chunk_to_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 - Range compression của chunk 13 và 14
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    CHUNK_CACHE_KEY,
    array_cache_matches,
    cache_fingerprint,
    compress_chunk_to_file,
    eta_13,
    input_identity,
    open_array,
    range_source,
    range_time_shift_13,
    raw_correction_source,
    raw_tau_13,
    save_cache_fingerprint,
    transmitted_pulse_samples,
):
    _directory = f"{CACHE_ROOT}/{CHUNK_CACHE_KEY}/range-compression-13"
    range_cache_13 = f"{_directory}/data.npy"
    _shape = (len(eta_13), len(raw_tau_13) - transmitted_pulse_samples + 1)
    _fingerprint = cache_fingerprint(
        "pair-range-v1", input_identity, CHUNK_CACHE_KEY, 13,
        range_time_shift_13, raw_correction_source, range_source,
    )
    if not array_cache_matches(_directory, range_cache_13, _fingerprint, _shape):
        tau_13, iq_bias_13 = compress_chunk_to_file(
            13, raw_tau_13, range_time_shift_13, range_cache_13
        )
        save_cache_fingerprint(_directory, _fingerprint)
    else:
        tau_13 = raw_tau_13[:_shape[1]] + range_time_shift_13
        iq_bias_13 = (float("nan"), float("nan"))
    range_compressed_13 = open_array(range_cache_13)
    return iq_bias_13, range_cache_13, range_compressed_13, tau_13


@app.cell
def _(
    CACHE_ROOT,
    CHUNK_CACHE_KEY,
    array_cache_matches,
    cache_fingerprint,
    compress_chunk_to_file,
    eta_14,
    input_identity,
    open_array,
    range_source,
    range_time_shift_14,
    raw_correction_source,
    raw_tau_14,
    save_cache_fingerprint,
    transmitted_pulse_samples,
):
    _directory = f"{CACHE_ROOT}/{CHUNK_CACHE_KEY}/range-compression-14"
    range_cache_14 = f"{_directory}/data.npy"
    _shape = (len(eta_14), len(raw_tau_14) - transmitted_pulse_samples + 1)
    _fingerprint = cache_fingerprint(
        "pair-range-v1", input_identity, CHUNK_CACHE_KEY, 14,
        range_time_shift_14, raw_correction_source, range_source,
    )
    if not array_cache_matches(_directory, range_cache_14, _fingerprint, _shape):
        tau_14, iq_bias_14 = compress_chunk_to_file(
            14, raw_tau_14, range_time_shift_14, range_cache_14
        )
        save_cache_fingerprint(_directory, _fingerprint)
    else:
        tau_14 = raw_tau_14[:_shape[1]] + range_time_shift_14
        iq_bias_14 = (float("nan"), float("nan"))
    range_compressed_14 = open_array(range_cache_14)
    return iq_bias_14, range_cache_14, range_compressed_14, tau_14


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 - Căn lưới range và ghép dải azimuth

    Phần lẻ SWST đã được sửa bằng phase ramp trong RRF. Phần nguyên được
    đặt vào lưới range hợp của hai chunk bằng black-fill, không nội suy lại ảnh.
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    CHUNK_CACHE_KEY,
    Path,
    az_sample_freq,
    doppler_centroid_estimation,
    doppler_source,
    eta_13,
    eta_14,
    input_identity,
    mo,
    np,
    open_array,
    range_cache_13,
    range_cache_14,
    tau_13,
    tau_14,
):
    def make_segments():
        return [
            doppler_centroid_estimation.Segment(
                open_array(range_cache_13), tau_13, eta_13, name="chunk 13"
            ),
            doppler_centroid_estimation.Segment(
                open_array(range_cache_14), tau_14, eta_14, name="chunk 14"
            ),
        ]

    def _combine_to_file(destination):
        _prepared = doppler_centroid_estimation.prepare_segments(
            make_segments(), prf_hz=az_sample_freq
        )
        _path = Path(destination)
        _path.parent.mkdir(parents=True, exist_ok=True)
        _temporary = _path.with_suffix(f"{_path.suffix}.tmp")
        _output = np.lib.format.open_memmap(
            _temporary,
            mode="w+",
            dtype=np.complex64,
            shape=(_prepared.num_azimuth_lines, _prepared.num_range_samples),
        )
        _prepared.align_into(_output, batch_lines=128)
        _output.flush()
        del _output
        _temporary.replace(_path)
        return (
            _prepared.common_slant_range_times_s,
            _prepared.azimuth_times_s,
            _prepared.alignment_summary(),
        )

    with mo.persistent_cache(
        name="range-aligned",
        save_path=f"{CACHE_ROOT}/{CHUNK_CACHE_KEY}",
        pin_modules=True,
    ):
        input_identity, doppler_source, range_cache_13, range_cache_14
        combined_range_cache = (
            f"{CACHE_ROOT}/{CHUNK_CACHE_KEY}/range-aligned/data.npy"
        )
        common_tau, combined_eta, alignment_summary = _combine_to_file(
            combined_range_cache
        )
    combined_range = open_array(combined_range_cache)
    chunk_boundary_line = len(eta_13)
    return (
        alignment_summary,
        chunk_boundary_line,
        combined_eta,
        combined_range,
        combined_range_cache,
        common_tau,
        make_segments,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 - Doppler centroid và effective velocity
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    CHUNK_CACHE_KEY,
    PRI,
    az_sample_freq,
    doppler_centroid_estimation,
    doppler_source,
    input_identity,
    make_segments,
    mo,
    raw_tau_13,
    s6_parameters,
):
    doppler_estimator = doppler_centroid_estimation.Estimator.for_stripmap_s6(
        prf_hz=az_sample_freq
    )

    def _estimate_doppler():
        _segments = make_segments()
        _start = float(_segments[0].azimuth_times_s[0])
        _stop = float(_segments[-1].azimuth_times_s[-1] + PRI)
        _estimates = doppler_estimator.estimate_segments(
            _segments,
            dce_range_start_s=float(raw_tau_13[0]),
            known_ambiguity_number=s6_parameters.DCE_AMBIGUITY_NUMBER,
            slice_start_times_s=[_start],
            last_slice_stop_time_s=_stop,
            product_start_time_s=_start,
            product_stop_time_s=_stop,
            zero_dop_minus_acq_time_s=0.0,
        )
        return _estimates

    with mo.persistent_cache(
        name="doppler-centroid",
        save_path=f"{CACHE_ROOT}/{CHUNK_CACHE_KEY}",
        pin_modules=True,
    ):
        input_identity, doppler_source
        doppler_estimates = _estimate_doppler()
    return doppler_estimator, doppler_estimates


@app.cell
def _(
    combined_eta,
    common_tau,
    doppler_estimator,
    doppler_estimates,
    effective_velocity,
    l0file,
    wavelength_m,
):
    def doppler_centroid_for_line(line_index):
        return doppler_estimator.evaluate_at_line(
            doppler_estimates,
            line_index=line_index,
            azimuth_times_s=combined_eta,
            slant_range_times_s=common_tau,
        )

    velocity_estimator = effective_velocity.Estimator.from_level0_product(
        l0file, wavelength_m
    )
    # Chunk 14 ends 0.90 s after the last embedded state-vector epoch.
    velocity_estimator.validate_time_coverage(
        combined_eta, max_extrapolation_s=1.0
    )
    return doppler_centroid_for_line, velocity_estimator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 - Focus toàn bộ chunk 13–14

    Kết quả là internal SLC trên support hợp lệ; chưa áp dụng post-processing,
    calibration và SAFE formatting của Level-1.
    """)
    return


@app.cell
def _(
    az_sample_freq,
    azimuth_processing,
    c,
    combined_eta,
    common_tau,
    doppler_centroid_for_line,
    s6_parameters,
    velocity_estimator,
    wavelength_m,
):
    FOCUS_FFT_LEN = s6_parameters.FOCUS_FFT_LENGTH
    AZIMUTH_PROCESSING_BANDWIDTH_HZ = s6_parameters.FOCUS_AZIMUTH_BANDWIDTH_HZ
    slant_ranges_m = common_tau * c / 2.0
    focus_layout = azimuth_processing.processing_blocks.calculate_layout(
        len(combined_eta),
        slant_ranges_m,
        combined_eta,
        doppler_centroid_for_line,
        velocity_estimator,
        wavelength_m=wavelength_m,
        azimuth_sample_frequency_hz=az_sample_freq,
        processing_bandwidth_hz=AZIMUTH_PROCESSING_BANDWIDTH_HZ,
        fft_length=FOCUS_FFT_LEN,
        extra_overlap_samples=s6_parameters.EXTRA_AZIMUTH_OVERLAP_SAMPLES,
    )
    output_geometry = azimuth_processing.processing_blocks.L1OutputGeometry.from_focus_support(
        combined_eta,
        len(slant_ranges_m),
        focus_layout,
        azimuth_sample_period_s=1.0 / az_sample_freq,
    )
    return (
        AZIMUTH_PROCESSING_BANDWIDTH_HZ,
        FOCUS_FFT_LEN,
        focus_layout,
        output_geometry,
        slant_ranges_m,
    )


@app.cell
def _(
    AZIMUTH_PROCESSING_BANDWIDTH_HZ,
    CACHE_ROOT,
    CHUNK_CACHE_KEY,
    FOCUS_FFT_LEN,
    PRI,
    Path,
    azimuth_processing,
    c,
    combined_eta,
    combined_range_cache,
    doppler_centroid_for_line,
    doppler_estimates,
    focus_layout,
    focus_source,
    input_identity,
    mo,
    np,
    open_array,
    output_geometry,
    range_sample_freq,
    range_sample_period,
    s6_parameters,
    slant_ranges_m,
    velocity_estimator,
    wavelength_m,
):
    def _focus_to_file(destination):
        _source = open_array(combined_range_cache)
        _path = Path(destination)
        _path.parent.mkdir(parents=True, exist_ok=True)
        _temporary = _path.with_suffix(f"{_path.suffix}.tmp")
        _output = np.lib.format.open_memmap(
            _temporary,
            mode="w+",
            dtype=np.complex64,
            shape=output_geometry.shape,
        )
        azimuth_processing.processing_blocks.focus_slc(
            _source,
            slant_ranges_m,
            combined_eta,
            doppler_centroid_for_line,
            velocity_estimator,
            focus_layout,
            wavelength_m=wavelength_m,
            speed_of_light_mps=c,
            azimuth_sample_period_s=PRI,
            range_sample_period_s=range_sample_period,
            range_sample_frequency_hz=range_sample_freq,
            processing_bandwidth_hz=AZIMUTH_PROCESSING_BANDWIDTH_HZ,
            fft_length=FOCUS_FFT_LEN,
            azimuth_time_correction_s=0.0,
            src_segment_samples=s6_parameters.SRC_SEGMENT_SAMPLES,
            rcmc_kernel_length=s6_parameters.RCMC_KERNEL_LENGTH,
            rcmc_phases=s6_parameters.RCMC_PHASES,
            output_geometry=output_geometry,
            output=_output,
        )
        _output.flush()
        del _output
        _temporary.replace(_path)

    with mo.persistent_cache(
        name="focused-merged",
        save_path=f"{CACHE_ROOT}/{CHUNK_CACHE_KEY}",
        pin_modules=True,
    ):
        input_identity, focus_source, doppler_estimates, combined_range_cache
        focused_cache_file = (
            f"{CACHE_ROOT}/{CHUNK_CACHE_KEY}/focused-merged/data.npy"
        )
        _focus_to_file(focused_cache_file)
    focused_slc = open_array(focused_cache_file)
    return focused_cache_file, focused_slc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 - Kết quả SLC đã ghép
    """)
    return


@app.cell
def _(
    alignment_summary,
    chunk_boundary_line,
    focused_cache_file,
    focused_slc,
    focus_layout,
    iq_bias_13,
    iq_bias_14,
):
    print("SLC shape:", focused_slc.shape)
    print("Đường biên chunk 13/14 tại azimuth line:", chunk_boundary_line)
    print("Cache SLC:", focused_cache_file)
    print("I/Q bias chunk 13 / 14:", iq_bias_13, iq_bias_14)
    print("Overlap focus:", focus_layout.overlap_samples, "lines")
    alignment_summary
    return


@app.cell
def _(chunk_boundary_line, colors, focused_slc, np, plt):
    _amplitude = np.abs(focused_slc[::20, ::20])
    _positive = _amplitude[_amplitude > 0]
    _vmin = np.percentile(_positive, 5)
    _vmax = np.percentile(_positive, 99.8)

    plt.figure(figsize=(12, 12), dpi=75)
    plt.title("Focused SLC — chunk 13 + 14")
    plt.imshow(
        _amplitude,
        origin="lower",
        cmap="gray",
        norm=colors.LogNorm(vmin=_vmin, vmax=_vmax),
        aspect="auto",
    )
    plt.axhline(chunk_boundary_line / 20, color="tab:red", linewidth=0.8)
    plt.xlabel("Slant range (mỗi 20 samples)")
    plt.ylabel("Azimuth (mỗi 20 lines)")
    plt.show()
    return


@app.cell
def _(focused_slc, notebook_started_at, perf_counter):
    focused_slc
    print(f"Total runtime: {perf_counter() - notebook_started_at:.2f} s")
    return


if __name__ == "__main__":
    app.run()
