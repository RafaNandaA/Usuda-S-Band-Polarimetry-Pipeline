# Frequency-resolved catalogue writer. Includes the per-pulse kernel (kernel.jl) and adds the archive writer that also
# stores per-channel integrated Stokes (/freq_resolved) and a constant V/I zero-point column. Loaded with include() into
# its own module by calibrate/apply.jl, which calls write_archive(rows, profs, freqs, tag, outdir).

include(joinpath(@__DIR__, "kernel.jl"))

# Channel centre frequencies [MHz], subband-major, the row order of the /freq_resolved datasets.
function channel_freqs_MHz()
    F = Float64[]
    for s = 1:n_subbands, c = 1:NCHAN
        push!(F, fmin_sub[s] + c*(fs/M_chan/1e6))
    end
    return F
end

# HDF5.jl writes column-major and h5py reads row-major, so every 2-D dataset is built (NF, Npulse) in Julia and read as
# (Npulse, NF) in Python, one pulse per row.
function write_archive(rows, profs, freqs, tag, outdir)
    isempty(profs) && (@printf("no profiles to archive\n"); return)
    Np = length(profs); Ns = length(profs[1].I)
    NF = length(freqs[1].I_on)
    path = joinpath(outdir, "CrabGRP_$(tag).h5")
    rr(s) = [getfield(p, s) for p in rows]
    pp(s) = [getfield(p, s) for p in profs]
    h5open(path, "w") do f
        c = create_group(f, "catalog")
        function put(name, vec, units)
            c[name] = vec                              # handles String and numeric vectors
            attributes(c[name])["units"] = units
        end
        put("nseq",        [string(x.nseq) for x in rows], "")
        put("utc",         [x.utc===nothing ? "" : Dates.format(x.utc,"yyyy-mm-ddTHH:MM:SS.sss") for x in rows], "ISO8601")
        put("sod_utc_s",   rr(:sod),    "s")
        put("phase",       rr(:phase),  "rotations")
        put("MP_IP_flag",  [ (x.phase<0.15||x.phase>0.85) ? 0 : (0.30<=x.phase<=0.55 ? 1 : -1) for x in rows], "0=MP,1=IP,-1=other")
        put("valid",       Int.(rr(:valid)), "bool")
        put("snr",         rr(:snr),    "ratio")
        put("detsnr",      rr(:detsnr), "ratio")
        put("boxcar_width",rr(:boxw),   "bins")
        put("boxcar_dt_us",rr(:boxus),  "us")
        put("width_us",    pp(:Won_us), "us")
        put("W_eq_us",     pp(:Weq_us), "us")
        # peak (single brightest bin, W = 1), a fixed-aperture reference
        for (nm,k,u) in (("I_peak",:Ipk,"calibrated_arb"),("Q_peak",:Qpk,"calibrated_arb"),
                         ("U_peak",:Upk,"calibrated_arb"),("V_peak",:Vpk,"calibrated_arb"),
                         ("L_peak",:Lpk,"calibrated_arb"),("P_peak",:Ppk,"calibrated_arb"),
                         ("LI_peak",:LIpk,"percent"),("VI_peak",:VIpk,"percent"),("PI_peak",:PIpk,"percent"),
                         ("LI_peak_err",:LIpk_e,"percent"),("VI_peak_err",:VIpk_e,"percent"),("PI_peak_err",:PIpk_e,"percent"),
                         ("PA_peak",:PApk,"deg"),("PA_err_peak",:PApk_e,"deg"))
            put(nm, pp(k), u)
        end
        # whole-pulse (Stokes summed over the 3σ on-pulse region)
        for (nm,k,u) in (("I_int",:Iin,"calibrated_arb"),("Q_int",:Qin,"calibrated_arb"),
                         ("U_int",:Uin,"calibrated_arb"),("V_int",:Vin,"calibrated_arb"),
                         ("L_int",:Lin,"calibrated_arb"),("P_int",:Pin,"calibrated_arb"),
                         ("LI_int",:LIin,"percent"),("VI_int",:VIin,"percent"),("PI_int",:PIin,"percent"),
                         ("LI_int_err",:LIin_e,"percent"),("VI_int_err",:VIin_e,"percent"),("PI_int_err",:PIin_e,"percent"),
                         ("PA_int",:PAin,"deg"),("PA_err_int",:PAin_e,"deg"))
            put(nm, pp(k), u)
        end
        # union aperture beside the 3-sigma columns, and both apertures' sample ranges
        for (nm,k,u) in (("I_int_union",:Iun,"calibrated_arb"),("Q_int_union",:Qun,"calibrated_arb"),
                         ("U_int_union",:Uun,"calibrated_arb"),("V_int_union",:Vun,"calibrated_arb"),
                         ("L_int_union",:Lun,"calibrated_arb"),("P_int_union",:Pun,"calibrated_arb"),
                         ("LI_int_union",:LIun,"percent"),("VI_int_union",:VIun,"percent"),("PI_int_union",:PIun,"percent"),
                         ("LI_int_union_err",:LIun_e,"percent"),("VI_int_union_err",:VIun_e,"percent"),("PI_int_union_err",:PIun_e,"percent"),
                         ("PA_int_union",:PAun,"deg"),("PA_err_int_union",:PAun_e,"deg"),
                         ("sigma_I_int_union",:sIun,"calibrated_arb"),("sigma_QU_int_union",:sQUun,"calibrated_arb"),
                         ("sigma_V_int_union",:sVun,"calibrated_arb"),("width_union_us",:Wun_us,"us"))
            put(nm, pp(k), u)
        end
        put("union_flag", pp(:union_flag), "runs merged (0 = union identical to the 3-sigma aperture)")
        for k in (:peak_bin, :ap3_lo, :ap3_hi, :union_lo, :union_hi)
            put(string(k), pp(k), "1-based 1-us bin of the full window profile; raw_profiles 0-based index = $(RAW_HALFWIN) + bin - peak_bin")
        end
        attributes(c["union_flag"])["rule"] = "union aperture: 3-sigma runs with peak >= $(UNION_PK_NSIG) sigma within +-$(UNION_WMAX_US) us " *
            "of the main peak merged, gaps included"
        put("pulse_fluence", pp(:Iin),    "calibrated_arb")   # = I_int (instrumental units)
        put("pulse_energy",  pp(:Iin),    "calibrated_arb")
        put("PA_std",        pp(:PA_std), "deg")
        put("PA_range",      pp(:PA_rng), "deg")
        put("L_snr",         pp(:Lsnr_pk),"ratio")
        put("V_snr",         pp(:Vsnr_in),"ratio")
        # off-pulse noise -> lets the analysis redo the debias and background subtraction
        put("sigma_I",  pp(:sI),  "calibrated_arb")
        put("sigma_QU", pp(:sQU), "calibrated_arb")
        put("sigma_V",  pp(:sV),  "calibrated_arb")
        put("baseline_mu_I", pp(:mI), "calibrated_arb")

        # ---- constant V/I zero point (pedestal) ----
        # The nebula is taken to have V = 0, so the median off-pulse V_neb/I_neb over the valid windows is the
        # instrumental circular zero point. It is stored with a corrected V column (V -> V - pedestal I), and V_int
        # itself is left as measured so the correction is never applied twice.
        vped_vals = Float64[]
        for r in rows
            (r.valid && isfinite(r.V_neb) && isfinite(r.I_neb) && r.I_neb != 0) &&
                push!(vped_vals, r.V_neb / r.I_neb)
        end
        V_pedestal = isempty(vped_vals) ? 0.0 : median(vped_vals)
        put("V_int_pedcorr",  [profs[i].Vin  - V_pedestal*profs[i].Iin for i in 1:Np], "calibrated_arb")
        put("VI_int_pedcorr", [profs[i].VIin - 100.0*V_pedestal        for i in 1:Np], "percent")
        attributes(c["V_int_pedcorr"])["note"] = "V_int - V_pedestal*I_int; V_pedestal=median(V_neb/I_neb)"

        # ---- /nebula_diag : off-pulse nebula Stokes and geometry per window ----
        # PA_neb_deg follows chi_BWG_deg (slope 1) when the calibration is right.
        nd = create_group(f, "nebula_diag")
        function putnd(name, vec, units)
            nd[name] = vec; attributes(nd[name])["units"] = units
        end
        putnd("sod_utc_s",   rr(:sod),     "s")
        putnd("q_deg",       rr(:q_deg),   "deg")
        putnd("el_deg",      rr(:el_deg),  "deg")
        putnd("az_deg",      rr(:az_deg),  "deg")
        putnd("chi_BWG_deg", rr(:chi_bwg), "deg")
        putnd("PA_neb_deg",  rr(:PA_neb),  "deg")
        putnd("L_neb",       rr(:L_neb),   "calibrated_arb")
        putnd("I_neb",       rr(:I_neb),   "calibrated_arb")
        putnd("V_neb",       rr(:V_neb),   "calibrated_arb")
        putnd("PA_neb_preD_deg", rr(:PA_neb_preD), "deg")
        putnd("L_neb_preD",      rr(:L_neb_preD),  "calibrated_arb")
        putnd("V_neb_preD",      rr(:V_neb_preD),  "calibrated_arb")
        attributes(nd)["note"] = "Off-pulse (nebula-dominated) band-summed Stokes per pulse window. " *
            "PA_neb_deg, L_neb, V_neb have the full calibration (D-terms and ellipticity applied), the same as the " *
            "pulse profiles. PA_neb_deg = 0.5*atan(U,Q) follows chi_BWG_deg = q + $(BWG_ALPHA)*El + $(BWG_GAMMA)*Az. " *
            "The *_preD fields are taken before the leakage stage (delay, phase and RM only), so V_neb_preD still " *
            "carries the (Q,U) -> V leak."

        # ---- /raw_profiles : Np×Ns, calibrated, NOT bgsub, NOT debiased, gzip ----
        rp = create_group(f, "raw_profiles")
        function putprof(name, key)
            # Store sample-major (Ns x Np). HDF5.jl + h5py reverse 2-D dimension order
            # (Julia column-major vs NumPy row-major), so a Julia (Ns,Np) array is read
            # in Python as (Npulse, Nsamp) with each pulse a row -- which is what the
            # analysis expects. (Storing (Np,Ns) here makes h5py see it transposed.)
            M = Array{Float64}(undef, Ns, Np)
            @inbounds for i in 1:Np; M[:, i] .= getfield(profs[i], key); end
            d = create_dataset(rp, name, Float64, (Ns, Np); chunk=(Ns, 1), deflate=4)
            write(d, M); attributes(d)["units"] = "calibrated_arb"
        end
        putprof("I", :I); putprof("Q", :Q); putprof("U", :U); putprof("V", :V)
        rp["peak_index"] = pp(:ipk_rel)
        attributes(rp["peak_index"])["units"] = "bin (0-based) of peak within each row"
        attributes(rp)["note"] = "calibrated Stokes; cropped to +/-$(RAW_HALFWIN) samples around the peak; " *
                                 "NOT baseline-subtracted; NOT debiased. Off-pulse = samples away from " *
                                 "peak_index on either side; per-pulse mu/sigma (from the full pre-crop " *
                                 "window) are in /catalog."

        # ---- /freq_resolved : per-channel integrated Stokes (Np x NF as read by h5py) ----
        # After the D-terms and before ellipticity and RM (feed frame), so the per-channel ellipticity and any RM can be
        # applied downstream. *_on = 3-sigma aperture, baseline-subtracted per channel, *_off = off-pulse (nebula) mean
        # per channel, *_union = union aperture. Channels are subband-major (freq_MHz). Written (NF, Np) in Julia, read
        # (Np, NF) in Python.
        fr = create_group(f, "freq_resolved")
        fr["freq_MHz"] = channel_freqs_MHz()
        attributes(fr["freq_MHz"])["units"] = "MHz"
        function putfreq(name, key, units)
            M = Array{Float64}(undef, NF, Np)
            @inbounds for i in 1:Np; M[:, i] .= getfield(freqs[i], key); end
            d = create_dataset(fr, name, Float64, (NF, Np); chunk=(NF, 1), deflate=4)
            write(d, M); attributes(d)["units"] = units
        end
        putfreq("I_chan",     :I_on,  "calibrated_arb")
        putfreq("Q_chan",     :Q_on,  "calibrated_arb")
        putfreq("U_chan",     :U_on,  "calibrated_arb")
        putfreq("V_chan",     :V_on,  "calibrated_arb")
        putfreq("I_off_chan", :I_off, "calibrated_arb")
        putfreq("Q_off_chan", :Q_off, "calibrated_arb")
        putfreq("U_off_chan", :U_off, "calibrated_arb")
        putfreq("V_off_chan", :V_off, "calibrated_arb")
        for (nm, k) in (("I_chan_union", :I_on_union), ("Q_chan_union", :Q_on_union),
                        ("U_chan_union", :U_on_union), ("V_chan_union", :V_on_union))
            putfreq(nm, k, "calibrated_arb")
        end
        attributes(fr)["note"] = "Per-channel integrated Stokes after the D-terms and before ellipticity and RM (feed " *
            "frame). *_chan = on-pulse (3-sigma aperture, baseline-subtracted with the per-channel first-20% mean); " *
            "*_chan_union = the same on the union aperture; *_off_chan = off-pulse (nebula) mean per channel. Row " *
            "order matches freq_MHz (subband-major). Apply the per-channel ellipticity (V -= m_QV*Q + m_UV*U) and RM " *
            "here to complete the calibration at frequency resolution. The band-summed /catalog has them applied."

        # ---- /metadata ----
        md = create_group(f, "metadata"); A = attributes(md)
        A["source_name"] = "Crab Pulsar"; A["telescope"] = "Usuda 64m"; A["backend"] = "S-band 2-pol"
        A["center_frequency_MHz"] = 2258.0; A["bandwidth_MHz"] = 128.0
        A["n_freq_channels"] = NF
        A["V_pedestal_VI"] = V_pedestal
        A["V_pedestal_note"] = "median off-pulse V_neb/I_neb; catalog V_int is as measured, V_int_pedcorr has it subtracted"
        A["time_resolution_us"] = TS_NAVG / fs * 1e6
        A["provenance"] = provenance_string()
        A["DM_pc_cm3"] = DM
        A["delay_method"] = "R/L delay as a sky-frequency phase ramp on L before channelisation"
        A["polarization_calibrated"] = "bandpass+gain+diffgain(t)+R/L phase+delay+D-term(kappa)+ellipticity(Q,U->V)"
        A["parallactic_corrected"] = "catalog PA_* are in the feed frame; sky PA = wrap90(PA - nebula_diag/chi_BWG_deg)"
        A["debias_method"] = "Wardle-Kronberg / Everett-Weisberg 2001 (catalog only; raw_profiles undebiased)"
        A["normalization_method"] = "catalog: first-20% baseline subtracted; raw_profiles: none"
        A["software_version"] = "usuda-polarimetry (frequency-resolved)"; A["creation_date"] = string(Dates.now())
    end
    @printf("hdf5 -> %s   (%d pulses, %d-sample profiles)\n", path, Np, Ns)
end
