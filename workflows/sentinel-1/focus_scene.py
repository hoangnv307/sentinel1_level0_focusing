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
    # Focus a complete Sentinel-1 Level-0 scene

    Processing flow: validate downlink headers, correct raw I/Q data, perform
    range compression, align the range grid, estimate Doppler centroid, and
    focus the complete scene into an SLC image.
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

    import notebook_support.cache as cache

    plt.style.use("default")
    return (
        PROJECT_ROOT,
        cache,
        colors,
        np,
        plt,
        sentinel1decoder,
    )


@app.cell(hide_code=True)
def _(cache):
    import sentinel1_processing.azimuth_pre_processing as azimuth_pre_processing
    import sentinel1_processing.azimuth_processing as azimuth_processing
    import sentinel1_processing.common as common
    import sentinel1_processing.doppler_centroid as doppler_centroid
    import sentinel1_processing.pre_processing.downlink_header_validation as downlink_header_validation
    import sentinel1_processing.range_processing as range_processing
    import sentinel1_processing.s6_parameters as s6_parameters

    range_source = cache.source_snapshot(
        azimuth_pre_processing,
        downlink_header_validation,
        range_processing,
        s6_parameters,
    )
    raw_correction_source = cache.source_snapshot(common.raw_data_correction)
    doppler_source = cache.source_snapshot(doppler_centroid, s6_parameters)
    focus_source = cache.source_snapshot(
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
    ## 1 - Input data and radar parameters
    """)
    return


@app.cell
def _(PROJECT_ROOT, cache, mo, sentinel1decoder):
    CACHE_ROOT = str(PROJECT_ROOT / ".cache" / "sentinel1")
    _input_path = (
        PROJECT_ROOT
        / "data"
        / "sao_paulo"
        / "s1a-s6-raw-s-vv-20251226t214356-20251226t214427-062491-07d496.dat"
    )
    _watched_input = mo.watch.file(str(_input_path))
    input_identity = cache.file_identity(_watched_input)
    l0file = sentinel1decoder.Level0File(str(_input_path))
    return CACHE_ROOT, input_identity, l0file


@app.cell
def _(cache, l0file, s6_parameters, sentinel1decoder):
    # DAD §4.3: establish the downlink-header values required by later
    # processing stages. The scene spans two contiguous echo-data chunks in
    # this Level-0 product; downstream cells treat them as generic segments.
    SCENE_CHUNKS = (13, 14)
    SCENE_CACHE_KEY = cache.scene_cache_key(SCENE_CHUNKS)
    scene_metadata = tuple(
        l0file.get_acquisition_chunk_metadata(chunk) for chunk in SCENE_CHUNKS
    )

    # DAD §6 uses PRI for the azimuth sampling grid and Range Decimation for
    # the complex range sampling frequency. TXPSF, TXPRR, and TXPL describe
    # the transmitted chirp used to construct the Range Reference Function.
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
        # DAD §4.3 / PDU §3.2.5.12: translate SWL and Range Decimation into
        # the expected Stage-3 complex sample count, then reject inconsistent
        # packets before any signal processing is attempted.
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
        # DAD §6.2.2: Rank and SWST locate every decoded sample on the slant-
        # range time axis. DAD §6.1.3 then applies the instrument SWST bias.
        _raw_tau = (
            metadata["Rank"].iloc[0] * PRI
            + metadata["SWST"].iloc[0]
            + suppressed_data_time
            + np.arange(_count) / range_sample_freq
        )
        _tau = range_processing.swst_bias.correct(_raw_tau, SWST_BIAS_S)
        # Packet coarse and fine times form the slow-time/azimuth axis used by
        # DCE and all later azimuth-domain processing.
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
        # DAD §6.2.2.1: split an SWST displacement into integer and fractional
        # samples. The fractional part is applied as an RRF phase ramp during
        # compression; integer offsets are handled by black-fill alignment.
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
    # DAD §6.1.1: build the normalized matched filter from the transmitted
    # chirp and transform it to range frequency. The FFT length covers linear
    # convolution for the largest SWL present in the complete scene.
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
    cache,
    common,
    l0file,
    np,
    range_reference_function,
    range_sample_freq,
    transmitted_pulse_samples,
):
    def compress_chunk(chunk, raw_tau, range_time_shift_s, destination):
        _radar_data = l0file.get_acquisition_chunk_data(chunk)
        # DAD §9.2.1: estimate and remove the constant I/Q bias immediately
        # after BAQ decoding. Other optional §9.2 corrections are not applied
        # by this demo.
        _iq_bias = np.complex128(
            common.raw_data_correction.estimate_iq_bias(_radar_data)
        )
        _shape = (
            _radar_data.shape[0],
            _radar_data.shape[1] - transmitted_pulse_samples,
        )

        def _write(output):
            # DAD §6.2.2: zero-pad each range line, FFT it, multiply by the
            # RRF, inverse FFT, and discard the matched-filter transient.
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

        range_times = cache.write_memmap(destination, _shape, _write)
        return range_times, (float(_iq_bias.real), float(_iq_bias.imag))

    return (compress_chunk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 - Range compression
    """)
    return


@app.cell(hide_code=True)
def _(
    CACHE_ROOT,
    SCENE_CACHE_KEY,
    SCENE_CHUNKS,
    cache,
    compress_chunk,
    eta_segments,
    input_identity,
    range_source,
    range_time_shifts,
    raw_correction_source,
    raw_tau_segments,
    transmitted_pulse_samples,
):
    def _load_chunk(chunk, eta, raw_tau, range_time_shift):
        shape = (len(eta), len(raw_tau) - transmitted_pulse_samples)
        fingerprint = cache.cache_fingerprint(
            "scene-range-v3", input_identity, SCENE_CACHE_KEY, chunk,
            range_time_shift, raw_correction_source, range_source,
        )
        path, _, created = cache.load_or_create_array(
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
    ## 3 - Range-grid alignment and azimuth assembly

    The fractional SWST offset is corrected by the RRF phase ramp. Integer
    offsets are placed on the segments' common range grid using black fill,
    without resampling the complex measurements.
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    SCENE_CACHE_KEY,
    az_sample_freq,
    cache,
    doppler_centroid,
    doppler_source,
    eta_segments,
    input_identity,
    range_cache_files,
    range_time_segments,
):
    def make_segments():
        return [
            doppler_centroid.estimation.Segment(
                cache.open_array(path),
                range_times,
                azimuth_times,
                name=f"segment {index}",
            )
            for index, (path, range_times, azimuth_times) in enumerate(
                zip(range_cache_files, range_time_segments, eta_segments), start=1
            )
        ]

    def _combine_to_file(destination):
        # DAD §6.2.2.2: SWL and SWST may change between range lines. Map every
        # segment onto one common slant-range grid, using black fill for the
        # integer offsets rather than resampling the complex measurements.
        _prepared = doppler_centroid.estimation.prepare_segments(
            make_segments(),
            prf_hz=az_sample_freq,
        )
        cache.write_memmap(
            destination,
            (_prepared.num_azimuth_lines, _prepared.num_range_samples),
            lambda output: _prepared.align_into(output, batch_lines=128),
        )
        return (
            _prepared.common_slant_range_times_s,
            _prepared.azimuth_times_s,
            _prepared.alignment_summary(),
        )

    @cache.persistent(f"{CACHE_ROOT}/{SCENE_CACHE_KEY}")
    def _combine_scene(_input_identity, _doppler_source, _range_cache_files):
        combined_range_cache = (
            f"{CACHE_ROOT}/{SCENE_CACHE_KEY}/range-aligned/data.npy"
        )
        common_tau, combined_eta, alignment_summary = _combine_to_file(
            combined_range_cache
        )
        return combined_range_cache, common_tau, combined_eta, alignment_summary

    (
        combined_range_cache,
        common_tau,
        combined_eta,
        alignment_summary,
    ) = _combine_scene(
        input_identity,
        doppler_source,
        range_cache_files,
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
    ## 4 - Doppler centroid and effective velocity
    """)
    return


@app.cell
def _(
    CACHE_ROOT,
    SCENE_CACHE_KEY,
    PRI,
    az_sample_freq,
    cache,
    c,
    doppler_centroid,
    doppler_source,
    input_identity,
    l0file,
    make_segments,
    raw_tau_segments,
    wavelength_m,
):
    doppler_estimator = doppler_centroid.estimation.Estimator.for_stripmap_s6(
        prf_hz=az_sample_freq
    )
    geometry_dc_estimator = doppler_centroid.geometry.Estimator.from_level0_product(
        l0file, wavelength_m
    )

    @cache.persistent(f"{CACHE_ROOT}/{SCENE_CACHE_KEY}")
    def _estimate_doppler(_input_identity, _doppler_source):
        # DAD §5.2-§5.5: estimate fine DC with lag-one correlation, unwrap it,
        # resolve the PRF ambiguity against the §5.1 orbit/attitude geometry
        # estimate, then fit the range-dependent Doppler polynomials.
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

    doppler_estimates = _estimate_doppler(
        input_identity,
        doppler_source,
    )
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

    # DAD §9.10: derive the effective radar velocity from the orbit state
    # vectors. The same range-dependent model feeds SRC, RCMC, and azimuth
    # compression, so it is evaluated through one shared estimator.
    velocity_estimator = common.effective_velocity.Estimator.from_level0_product(
        l0file, wavelength_m
    )
    # The scene ends 0.90 s after the product's final state-vector epoch.
    velocity_estimator.validate_time_coverage(
        combined_eta, max_extrapolation_s=1.0
    )
    return doppler_centroid_for_line, velocity_estimator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 - Focus the complete scene

    The result is an internal SLC over the valid processing support. Level-1
    post-processing, radiometric calibration, and SAFE formatting are outside
    this workflow.
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
    # DAD §9.12-§9.13: choose the Stripmap focusing-block overlap and FFT
    # length from the azimuth matched-filter support over the whole scene.
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
    # DAD §6.3.2 and §8.3.1: trim azimuth-filter and RCMC edge transients and
    # anchor the valid SLC on its zero-Doppler output timeline.
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
    cache,
    c,
    combined_eta,
    combined_range_cache,
    doppler_centroid_for_line,
    doppler_estimates,
    focus_layout,
    focus_source,
    input_identity,
    output_geometry,
    range_sample_freq,
    range_sample_period,
    s6_parameters,
    slant_ranges_m,
    velocity_estimator,
    wavelength_m,
):
    def _focus_to_file(destination):
        _source = cache.open_array(combined_range_cache)

        def _write_focus(output):
            # DAD §6.2.1/§6.2.3 and §6.3: process overlapping azimuth blocks
            # through zero-padding, azimuth FFT, SRC (§6.3.1), RCMC (§6.3.2),
            # and azimuth matched filtering/compression (§6.3.4).
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

        cache.write_memmap(
            destination,
            output_geometry.shape,
            _write_focus,
        )

    @cache.persistent(f"{CACHE_ROOT}/{SCENE_CACHE_KEY}")
    def _focus_scene(
        _input_identity,
        _focus_source,
        _doppler_estimates,
        _combined_range_cache,
    ):
        focused_cache_file = (
            f"{CACHE_ROOT}/{SCENE_CACHE_KEY}/focused-scene/data.npy"
        )
        _focus_to_file(focused_cache_file)
        return focused_cache_file

    focused_cache_file = _focus_scene(
        input_identity,
        focus_source,
        doppler_estimates,
        combined_range_cache,
    )
    focused_slc = cache.open_array(focused_cache_file)
    return (focused_slc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 - Scene SLC result
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
