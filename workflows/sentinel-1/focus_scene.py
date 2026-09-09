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
    # Focus một scene Sentinel-1 Level-0 hoàn chỉnh

    Luồng chính: kiểm tra header, sửa I/Q, range compression, căn lưới range,
    ước lượng Doppler centroid và focus toàn bộ scene thành SLC.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np
    import sentinel1decoder

    from notebook_support.cache import (
        cache_fingerprint,
        file_identity,
        load_or_create_array,
        open_array,
        scene_cache_key,
        source_snapshot,
        write_memmap,
    )

    plt.style.use("default")
    return (
        PROJECT_ROOT,
        cache_fingerprint,
        colors,
        file_identity,
        load_or_create_array,
        np,
        open_array,
        plt,
        scene_cache_key,
        sentinel1decoder,
        source_snapshot,
        write_memmap,
    )


@app.cell
def _(source_snapshot):
    import sentinel1_processing.azimuth_pre_processing as azimuth_pre_processing
    import sentinel1_processing.azimuth_processing as azimuth_processing
    import sentinel1_processing.common as common
    import sentinel1_processing.doppler_centroid as doppler_centroid
    import sentinel1_processing.pre_processing.downlink_header_validation as downlink_header_validation
    import sentinel1_processing.range_processing as range_processing
    import sentinel1_processing.s6_parameters as s6_parameters

    range_source = source_snapshot(
        azimuth_pre_processing,
        downlink_header_validation,
        range_processing,
        s6_parameters,
    )
    raw_correction_source = source_snapshot(common.raw_data_correction)
    doppler_source = source_snapshot(doppler_centroid, s6_parameters)
    focus_source = source_snapshot(
        azimuth_processing, common.effective_velocity, s6_parameters
    )
    return (
        azimuth_pre_processing,
        azimuth_processing,
        common,
        doppler_centroid,
        doppler_source,
        downlink_header_validation,
        focus_source,
        range_processing,
        range_source,
        raw_correction_source,
        s6_parameters,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 - Dữ liệu vào và tham số radar
    """)
    return


@app.cell
def _(PROJECT_ROOT, file_identity, mo, sentinel1decoder):
    CACHE_ROOT = str(PROJECT_ROOT / ".cache" / "sentinel1")
    _input_path = (
        PROJECT_ROOT
        / "data"
        / "sao_paulo"
        / "s1a-s6-raw-s-vv-20251226t214356-20251226t214427-062491-07d496.dat"
    )
    _watched_input = mo.watch.file(str(_input_path))
    input_identity = file_identity(_watched_input)
    l0file = sentinel1decoder.Level0File(str(_input_path))
    return CACHE_ROOT, input_identity, l0file


@app.cell
def _(l0file, s6_parameters, scene_cache_key, sentinel1decoder):
    SCENE_CHUNKS = (13, 14)
    SCENE_CACHE_KEY = scene_cache_key(SCENE_CHUNKS)
    scene_metadata = tuple(
        l0file.get_acquisition_chunk_metadata(chunk) for chunk in SCENE_CHUNKS
    )

    c = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS
    wavelength_m = s6_parameters.RADAR_WAVELENGTH_M
    _first = scene_metadata[0]
    PRI = float(_first["PRI"].iloc[0])
    RGDEC = _first["Range Decimation"].iloc[0]
    TXPSF = float(_first["Tx Pulse Start Frequency"].iloc[0])
    TXPRR = float(_first["Tx Ramp Rate"].iloc[0])
    TXPL = float(_first["Tx Pulse Length"].iloc[0])
    range_sample_freq = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    range_sample_period = 1.0 / range_sample_freq
    az_sample_freq = 1.0 / PRI
    suppressed_data_time = 320.0 / (8.0 * sentinel1decoder.constants.F_REF)
    return (
        SCENE_CACHE_KEY,
        SCENE_CHUNKS,
        PRI,
        TXPL,
        TXPRR,
        TXPSF,
        az_sample_freq,
        c,
        range_sample_freq,
        range_sample_period,
        scene_metadata,
        suppressed_data_time,
        wavelength_m,
    )


@app.cell
def _(
    PRI,
    downlink_header_validation,
    np,
    range_processing,
    range_sample_freq,
    s6_parameters,
    sentinel1decoder,
    scene_metadata,
    suppressed_data_time,
):
    SWST_BIAS_S = s6_parameters.SWST_BIAS_S

    def _axes(metadata):
        _count = 2 * int(metadata["Number of Quads"].iloc[0])
        _swl_code = round(
            float(metadata["SWL"].iloc[0]) * sentinel1decoder.constants.F_REF
        )
        _expected = downlink_header_validation.sample_count.rgdec9_stage3_rx_samples(
            _swl_code
        )
        if _count != _expected:
            raise ValueError(
                f"PDU Stage-3 sample count is {_expected}, packet contains {_count}."
            )
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

    _axes_by_segment = tuple(_axes(metadata) for metadata in scene_metadata)
    raw_tau_segments, eta_segments, raw_range_counts = map(
        tuple, zip(*_axes_by_segment)
    )
    _common_start = min(tau[0] for tau in raw_tau_segments)

    def _fractional_shift(tau):
        _offset = (tau[0] - _common_start) * range_sample_freq
        return (round(_offset) - _offset) / range_sample_freq

    range_time_shifts = tuple(map(_fractional_shift, raw_tau_segments))
    return (
        eta_segments,
        range_time_shifts,
        raw_range_counts,
        raw_tau_segments,
    )


@app.cell
def _(
    TXPL,
    TXPRR,
    TXPSF,
    np,
    range_processing,
    range_sample_freq,
    raw_range_counts,
):
    transmitted_pulse_samples = int(np.ceil(TXPL * range_sample_freq))
    _max_raw_samples = max(raw_range_counts)
    range_fft_length = 1 << int(np.ceil(np.log2(
        _max_raw_samples + transmitted_pulse_samples - 1
    )))
    range_reference_function = range_processing.range_reference_function.create_freq_domain(
        sample_rate_hz=range_sample_freq,
        pulse_start_frequency_hz=TXPSF,
        pulse_ramp_rate_hz_per_s=TXPRR,
        pulse_length_s=TXPL,
        fft_length=range_fft_length,
    )
    return range_reference_function, transmitted_pulse_samples


@app.cell
def _(
    TXPL,
    TXPRR,
    TXPSF,
    azimuth_pre_processing,
    common,
    l0file,
    np,
    range_reference_function,
    range_sample_freq,
    transmitted_pulse_samples,
    write_memmap,
):
    def compress_chunk(chunk, raw_tau, range_time_shift_s, destination):
        _radar_data = l0file.get_acquisition_chunk_data(chunk)
        _iq_bias = np.complex128(
            common.raw_data_correction.estimate_iq_bias(_radar_data)
        )
        _shape = (
            _radar_data.shape[0],
            _radar_data.shape[1] - transmitted_pulse_samples,
        )

        def _write(output):
            _, range_times = azimuth_pre_processing.range.compression.compress(
                _radar_data,
                raw_tau,
                sample_rate_hz=range_sample_freq,
                pulse_start_frequency_hz=TXPSF,
                pulse_ramp_rate_hz_per_s=TXPRR,
                pulse_length_s=TXPL,
                iq_bias=_iq_bias,
                range_reference_function=range_reference_function,
                range_time_shift_s=range_time_shift_s,
                output="slc",
                output_array=output,
            )
            return range_times

        range_times = write_memmap(destination, _shape, _write)
        return range_times, (float(_iq_bias.real), float(_iq_bias.imag))

    return (compress_chunk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 - Range compression
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    SCENE_CACHE_KEY,
    SCENE_CHUNKS,
    cache_fingerprint,
    compress_chunk,
    eta_segments,
    input_identity,
    load_or_create_array,
    range_source,
    range_time_shifts,
    raw_correction_source,
    raw_tau_segments,
    transmitted_pulse_samples,
):
    def _load_chunk(chunk, eta, raw_tau, range_time_shift):
        shape = (len(eta), len(raw_tau) - transmitted_pulse_samples)
        fingerprint = cache_fingerprint(
            "scene-range-v3", input_identity, SCENE_CACHE_KEY, chunk,
            range_time_shift, raw_correction_source, range_source,
        )
        path, _, created = load_or_create_array(
            f"{CACHE_ROOT}/{SCENE_CACHE_KEY}/range-compression-{chunk}",
            fingerprint,
            shape,
            lambda destination: compress_chunk(
                chunk, raw_tau, range_time_shift, destination
            ),
        )
        if created is None:
            return (
                str(path),
                raw_tau[:shape[1]] + range_time_shift,
                (float("nan"),) * 2,
            )
        range_times, iq_bias = created
        return str(path), range_times, iq_bias

    _compressed_segments = tuple(
        _load_chunk(chunk, eta, raw_tau, range_time_shift)
        for chunk, eta, raw_tau, range_time_shift in zip(
            SCENE_CHUNKS,
            eta_segments,
            raw_tau_segments,
            range_time_shifts,
        )
    )
    range_cache_files, range_time_segments, iq_biases = map(
        tuple, zip(*_compressed_segments)
    )
    return iq_biases, range_cache_files, range_time_segments


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 - Căn lưới range và ghép dải azimuth

    Phần lẻ SWST đã được sửa bằng phase ramp trong RRF. Phần nguyên được
    đặt vào lưới range chung của các segment bằng black-fill, không nội suy
    lại ảnh.
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    SCENE_CACHE_KEY,
    az_sample_freq,
    doppler_centroid,
    doppler_source,
    eta_segments,
    input_identity,
    mo,
    open_array,
    range_cache_files,
    range_time_segments,
    write_memmap,
):
    def make_segments():
        return [
            doppler_centroid.estimation.Segment(
                open_array(path), range_times, azimuth_times, name=f"segment {index}"
            )
            for index, (path, range_times, azimuth_times) in enumerate(
                zip(range_cache_files, range_time_segments, eta_segments), start=1
            )
        ]

    def _combine_to_file(destination):
        _prepared = doppler_centroid.estimation.prepare_segments(
            make_segments(),
            prf_hz=az_sample_freq,
        )
        write_memmap(
            destination,
            (_prepared.num_azimuth_lines, _prepared.num_range_samples),
            lambda output: _prepared.align_into(output, batch_lines=128),
        )
        return (
            _prepared.common_slant_range_times_s,
            _prepared.azimuth_times_s,
            _prepared.alignment_summary(),
        )

    with mo.persistent_cache(
        name="range-aligned",
        save_path=f"{CACHE_ROOT}/{SCENE_CACHE_KEY}",
        pin_modules=True,
    ):
        input_identity, doppler_source, range_cache_files
        combined_range_cache = (
            f"{CACHE_ROOT}/{SCENE_CACHE_KEY}/range-aligned/data.npy"
        )
        common_tau, combined_eta, alignment_summary = _combine_to_file(
            combined_range_cache
        )
    return (
        alignment_summary,
        combined_eta,
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
    SCENE_CACHE_KEY,
    PRI,
    az_sample_freq,
    c,
    doppler_centroid,
    doppler_source,
    input_identity,
    l0file,
    make_segments,
    mo,
    raw_tau_segments,
    wavelength_m,
):
    doppler_estimator = doppler_centroid.estimation.Estimator.for_stripmap_s6(
        prf_hz=az_sample_freq
    )
    geometry_dc_estimator = doppler_centroid.geometry.Estimator.from_level0_product(
        l0file, wavelength_m
    )

    def _estimate_doppler():
        _segments = make_segments()
        _start = float(_segments[0].azimuth_times_s[0])
        _stop = float(_segments[-1].azimuth_times_s[-1] + PRI)
        _estimates = doppler_estimator.estimate_segments(
            _segments,
            dce_range_start_s=float(raw_tau_segments[0][0]),
            geometry_dc_provider=lambda time_s, range_times_s: (
                geometry_dc_estimator.estimate(time_s, range_times_s * c / 2.0)
            ),
            slice_start_times_s=[_start],
            last_slice_stop_time_s=_stop,
            product_start_time_s=_start,
            product_stop_time_s=_stop,
            zero_dop_minus_acq_time_s=0.0,
        )
        return _estimates

    with mo.persistent_cache(
        name="doppler-centroid",
        save_path=f"{CACHE_ROOT}/{SCENE_CACHE_KEY}",
        pin_modules=True,
    ):
        input_identity, doppler_source
        doppler_estimates = _estimate_doppler()
    return doppler_estimates, doppler_estimator, geometry_dc_estimator


@app.cell
def _(
    combined_eta,
    common,
    common_tau,
    doppler_estimates,
    doppler_estimator,
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

    velocity_estimator = common.effective_velocity.Estimator.from_level0_product(
        l0file, wavelength_m
    )
    # Scene kết thúc 0,90 s sau state-vector epoch cuối cùng trong sản phẩm.
    velocity_estimator.validate_time_coverage(
        combined_eta, max_extrapolation_s=1.0
    )
    return doppler_centroid_for_line, velocity_estimator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 - Focus toàn bộ scene

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
    geometry_dc_estimator,
    range_sample_freq,
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
    output_geometry = azimuth_processing.processing_blocks.derive_output_geometry(
        slant_ranges_m,
        combined_eta,
        doppler_centroid_for_line,
        lambda line: geometry_dc_estimator.estimate(
            combined_eta[line], slant_ranges_m
        ),
        velocity_estimator,
        focus_layout,
        wavelength_m=wavelength_m,
        speed_of_light_mps=c,
        azimuth_sample_period_s=1.0 / az_sample_freq,
        range_sample_frequency_hz=range_sample_freq,
        processing_bandwidth_hz=AZIMUTH_PROCESSING_BANDWIDTH_HZ,
        fft_length=FOCUS_FFT_LEN,
        rcmc_kernel_length=s6_parameters.RCMC_KERNEL_LENGTH,
        rcmc_phases=s6_parameters.RCMC_PHASES,
        apply_coarse_bistatic_delay_correction=True,
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
    SCENE_CACHE_KEY,
    FOCUS_FFT_LEN,
    PRI,
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
    open_array,
    output_geometry,
    range_sample_freq,
    range_sample_period,
    s6_parameters,
    slant_ranges_m,
    velocity_estimator,
    wavelength_m,
    write_memmap,
):
    def _focus_to_file(destination):
        _source = open_array(combined_range_cache)

        def _focus(output):
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
                output=output,
            )

        write_memmap(
            destination,
            output_geometry.shape,
            _focus,
        )

    with mo.persistent_cache(
        name="focused-scene",
        save_path=f"{CACHE_ROOT}/{SCENE_CACHE_KEY}",
        pin_modules=True,
    ):
        input_identity, focus_source, doppler_estimates, combined_range_cache
        focused_cache_file = (
            f"{CACHE_ROOT}/{SCENE_CACHE_KEY}/focused-scene/data.npy"
        )
        _focus_to_file(focused_cache_file)
    focused_slc = open_array(focused_cache_file)
    return (focused_slc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 - Kết quả SLC của scene
    """)
    return


@app.cell
def _(
    alignment_summary,
    focus_layout,
    focused_slc,
    iq_biases,
):
    print("SLC shape:", focused_slc.shape)
    print("I/Q bias các segment:", iq_biases)
    print("Overlap focus:", focus_layout.overlap_samples, "lines")
    alignment_summary
    return


@app.cell
def _(colors, focused_slc, np, plt):
    _amplitude = np.abs(focused_slc[::20, ::20])
    _positive = _amplitude[_amplitude > 0]
    _vmin = np.percentile(_positive, 5)
    _vmax = np.percentile(_positive, 99.8)

    plt.figure(figsize=(12, 12), dpi=75)
    plt.title("Focused SLC — toàn scene")
    plt.imshow(
        _amplitude,
        origin="lower",
        cmap="viridis",
        norm=colors.LogNorm(vmin=_vmin, vmax=_vmax),
        aspect="auto",
    )
    plt.xlabel("Slant range (mỗi 20 samples)")
    plt.ylabel("Azimuth (mỗi 20 lines)")
    plt.show()
    return


@app.cell
def _(colors, focused_slc, np, plt):
    _amplitude = np.abs(focused_slc[9000:10500, 5500:6800])
    _positive = _amplitude[_amplitude > 0]
    _vmin = np.percentile(_positive, 2.5)
    _vmax = np.percentile(_positive, 98)

    plt.figure(figsize=(12, 12), dpi=75)
    plt.title("Focused SLC — chi tiết")
    plt.imshow(
        _amplitude,
        origin="lower",
        cmap="viridis",
        norm=colors.LogNorm(vmin=_vmin, vmax=_vmax),
        aspect="auto",
    )
    plt.xlabel("Slant range")
    plt.ylabel("Azimuth")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
