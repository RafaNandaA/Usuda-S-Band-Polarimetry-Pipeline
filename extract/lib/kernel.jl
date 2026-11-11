# Per-pulse kernel: coherent dedispersion, channelisation, the calibration chain and the per-window measurement.
# Loaded with include() into its own module by the entry scripts (extract/extract.jl, calibrate/apply.jl). The module
# also writes the catalogue HDF5 (write_archive), used by calibrate/apply.jl.
#
# Stage 1: each subband and hand is dedispersed coherently with one long FFT over the whole read window (reference
#          f_ref_MHz). L's operator also carries the R/L delay as a sky-frequency phase ramp (params.jl).
# Stage 2: 64-point FFT of the dedispersed analytic signal (32 channels of 1 MHz, 1-us samples), then per channel the
#          calibration chain bandpass -> gain -> gamma(t) -> R/L phase -> residual delay -> D-terms -> Stokes
#          -> ellipticity -> RM, and the band sum with the channel weights.
# process_window(...; rawcoh=true) also returns the uncalibrated per-channel coherency (RR, LL, Re RL, Im RL) per
# subband, channel and 1-us sample, taken after stage 1 and before any calibration term.
#
# The boxcar matched filter is used for detection and S/N only. Polarization is measured at the peak sample and over
# the on-pulse apertures, so the detection window is never reused as the measurement aperture.

using FFTW, Statistics, Printf, Dates, HDF5, LinearAlgebra

include(joinpath(@__DIR__, "params.jl"))

const ONPULSE_NSIG = Float64(cfgget(CFG, "aperture", "onpulse_nsig", 3.0))    # on-pulse run = contiguous bins with (I-mu) > NSIG sigma

const ALPHA   = 4.148808e9
const C_MPS   = 2.99792458e8
const TS_NAVG = M_chan * N_ave
const NCHAN   = M_chan ÷ 2
const CPLAN   = plan_fft(zeros(ComplexF64, M_chan))
const WORD_BYTES = 4

# A window is valid when the detected peak lies in this 5-ms box (time axis of the read, which starts at t_start_s).
# Listed pulses are found 12.2 ms into the read (t = 0.067 s), so a peak outside the box is not the listed pulse.
const PK_LO = 0.064
const PK_HI = 0.069
# A listed pulse is kept when its larger per-hand list S/N reaches this value (lists are per hand, threshold 5).
const MIN_DET_SNR = Float64(cfgget(CFG, "selection", "min_det_snr", 6.0))
# Baseline = first fraction of the kept window (at least 10 samples).
const BASELINE_FRAC = Float64(cfgget(CFG, "aperture", "baseline_fraction", 0.2))

# Boxcar matched-filter widths (in 1-us bins), used for detection S/N only. W = 1 is the single-bin S/N.
const BOXCAR_WIDTHS = [1, 2, 4, 8, 16, 32, 64]

# Debias positive-definite linear/total polarisation (L = √(max(0, Q²+U²-σ²))).
# Removes the one-sided noise bias that pushes L/I, p/I above 100% at low S/N.
# Signed V is unbiased and never debiased. Set false to report raw L, p.
const DEBIAS_POL = true

# Half-width (samples) kept on each side of the peak in /raw_profiles. At 1 us per sample this is 750 us each side,
# wide enough for the on-pulse measurement and for re-estimating the off-pulse baseline downstream.
const RAW_HALFWIN = Int(cfgget(CFG, "aperture", "raw_profile_half_us", 750))

# Read placement. A pulse at list time sod is read from word round((sod - SOD_RAW_START) * fs): the file clock starts at
# 08:00:00 UTC (file name), and the list times sit 31.4 ms before the file clock.
const T_READ_START = 0.055   # window start on the read time axis (= t_start_s), so the two cancel in the word offset
const SOD_RAW_START  = 28799.9686

const USUDA_LAT_DEG = Float64(cfgget(CFG, "observation", "site_lat_deg", 36.1317))
const USUDA_LON_DEG = Float64(cfgget(CFG, "observation", "site_lon_deg", 138.3650))
const CRAB_RA_DEG   = 83.63308     # J2000
const CRAB_DEC_DEG  = 22.01446

# Feed angle chi = q + BWG_ALPHA El + BWG_GAMMA Az. The beam-waveguide feed room turns in azimuth and does not tip, so
# the feed sees the sky rotated by q - El (alpha = -1, no azimuth term). Used for the per-window nebula diagnostic.
const BWG_ALPHA = Float64(cfgget(CFG, "feed", "alpha_el", -1.0))
const BWG_GAMMA = Float64(cfgget(CFG, "feed", "gamma_az", 0.0))

@inline extract(word::UInt32, ch::Int)::Float64 =
    Float64((word >> (4 * ch)) & 0x0000000F) - 7.5

function dedisp_operator(Lwin::Int, f_lo::Float64)
    beta = zeros(ComplexF64, Lwin)
    Nnyq = Lwin ÷ 2
    df_MHz = fs / (Lwin * 1.0e6)
    sqrt2 = sqrt(2.0)
    @inbounds for k = 1:Nnyq-1
        f = f_lo + k * df_MHz
        ps = -ALPHA * DM / f * (f_ref_MHz - f)^2 / f_ref_MHz^2
        ps -= floor(ps)
        beta[k + 1] = sqrt2 * cis(2π * ps)
    end
    f = f_lo + Nnyq * df_MHz
    ps = -ALPHA * DM / f * (f_ref_MHz - f)^2 / f_ref_MHz^2
    ps -= floor(ps)
    beta[Nnyq + 1] = cis(2π * ps)
    return beta
end

dedisperse(x, beta, fp, ip) = ip * ((fp * complex(x)) .* beta)

# Stage 1 and 2 run with cached dedispersion operators, shared FFTW plans (ESTIMATE, built under a lock) and per-task
# workspaces, so many windows can be processed in parallel threads. Every sum runs in a fixed order, so the output does
# not depend on the number of threads.
const _KLOCK = ReentrantLock()
const _OPCACHE = Dict{Tuple{Int,Int,Float64},Tuple{Vector{ComplexF64},Vector{ComplexF64}}}()
"(beta, betaL) for (Lwin, subband s, current DM): betaL = beta times the R/L delay ramp."
function cached_operators(Lwin::Int, s::Int)
    key = (Lwin, s, Float64(DM))
    lock(_KLOCK) do
        get!(_OPCACHE, key) do
            beta = dedisp_operator(Lwin, fmin_sub[s])
            betaL = beta .* ramp_L(Lwin, fmin_sub[s], delay_RL[s])
            (beta, betaL)
        end
    end
end
clear_operator_cache!() = lock(() -> empty!(_OPCACHE), _KLOCK)

"per-task stage-1 workspace: xc (complex input), X (spectrum), zR, zL (dedispersed hands)."
struct Stage1WS
    xc::Vector{ComplexF64}; X::Vector{ComplexF64}; zR::Vector{ComplexF64}; zL::Vector{ComplexF64}
end
Stage1WS(L::Int) = Stage1WS(Vector{ComplexF64}(undef, L), Vector{ComplexF64}(undef, L),
                            Vector{ComplexF64}(undef, L), Vector{ComplexF64}(undef, L))
const _WSPOOL = Dict{Int,Vector{Stage1WS}}()
"borrow a workspace of length L for the duration of f(ws) (pool grows to the number of concurrent tasks)."
function with_ws(f, L::Int)
    ws = lock(_KLOCK) do
        v = get!(() -> Stage1WS[], _WSPOOL, L)
        isempty(v) ? Stage1WS(L) : pop!(v)
    end
    try
        return f(ws)
    finally
        lock(() -> push!(_WSPOOL[L], ws), _KLOCK)
    end
end
release_workspaces!() = lock(() -> empty!(_WSPOOL), _KLOCK)

"dedisperse R with beta and L with betaL (ramp included) into ws.zR and ws.zL."
function stage1_pair!(ws::Stage1WS, buf::AbstractVector{UInt32}, s::Int, beta::Vector{ComplexF64}, betaL::Vector{ComplexF64},
                      fp, ip)
    xc = ws.xc; X = ws.X; Lwin = length(buf)
    @inbounds for n = 1:Lwin; xc[n] = complex(extract(buf[n], s-1)); end
    mul!(X, fp, xc); X .*= beta; mul!(ws.zR, ip, X)
    @inbounds for n = 1:Lwin; xc[n] = complex(extract(buf[n], s-1+4)); end
    mul!(X, fp, xc); X .*= betaL; mul!(ws.zL, ip, X)
    return nothing
end

# Per-channel calibration and band sum for one subband (type-stable, globals passed as arguments).
function _stokes_kernel!(I, Q, U, V, Qnl, Unl, Vnl, zR, zL, rawcoh, use_rc::Bool,
                         ibpR::Vector{Float64}, ibpL::Vector{Float64}, use_bpR::Bool, use_bpL::Bool,
                         gR::Float64, gL::Float64, inv_gamma::Float64, ph::ComplexF64,
                         delayph::Vector{ComplexF64}, use_dph::Bool, chw::Vector{Float64},
                         dR_v::Vector{ComplexF64}, dL_v::Vector{ComplexF64}, do_leak::Bool,
                         mQV_v::Vector{Float64}, mUV_v::Vector{Float64}, do_vleak::Bool,
                         fcs::Vector{Float64}, fsn::Vector{Float64}, Mc::Int, Nav::Int, ntime::Int)
    xR = zeros(ComplexF64, Mc); xL = zeros(ComplexF64, Mc)
    Rk = zeros(ComplexF64, Mc); Lk = zeros(ComplexF64, Mc)
    for t = 1:ntime
        aI=0.0; aQ=0.0; aU=0.0; aV=0.0; aQnl=0.0; aUnl=0.0; aVnl=0.0
        for b = 1:Nav
            blk = ((t-1)*Nav + (b-1)) * Mc
            @inbounds for i = 1:Mc; xR[i] = zR[blk+i]; xL[i] = zL[blk+i]; end
            mul!(Rk, CPLAN, xR); mul!(Lk, CPLAN, xL)
            @inbounds for c = 1:NCHAN
                r = Rk[c+1]; l = Lk[c+1]
                if use_rc
                    rlu = r*conj(l)
                    rawcoh[1,c,t] += abs2(r); rawcoh[2,c,t] += abs2(l)
                    rawcoh[3,c,t] += real(rlu); rawcoh[4,c,t] += imag(rlu)
                end
                use_bpR && (r *= ibpR[c]); use_bpL && (l *= ibpL[c])
                r /= gR; l /= gL; l *= inv_gamma; l *= ph
                use_dph && (l *= delayph[c])
                wch = chw[c]
                rlnl = r*conj(l)
                Qcnl = 2*real(rlnl); Ucnl = 2*imag(rlnl)
                aQnl += wch*(fcs[c]*Qcnl + fsn[c]*Ucnl)
                aUnl += wch*(-fsn[c]*Qcnl + fcs[c]*Ucnl)
                aVnl += wch*(abs2(r) - abs2(l))
                if do_leak
                    r0 = r; r = r0 - dR_v[c]*l; l = l - dL_v[c]*r0
                end
                rl = r*conj(l)
                Ic = abs2(r) + abs2(l); Vc = abs2(r) - abs2(l)
                Qc = 2*real(rl);        Uc = 2*imag(rl)
                do_vleak && (Vc -= mQV_v[c]*Qc + mUV_v[c]*Uc)
                Qd =  fcs[c]*Qc + fsn[c]*Uc
                Ud = -fsn[c]*Qc + fcs[c]*Uc
                aI += wch*Ic; aV += wch*Vc; aQ += wch*Qd; aU += wch*Ud
            end
        end
        nrm = Nav * NCHAN
        if use_rc && Nav != 1
            @inbounds for c = 1:NCHAN, k = 1:4; rawcoh[k,c,t] /= Nav; end
        end
        @inbounds begin
            I[t]=aI/nrm; Q[t]=aQ/nrm; U[t]=aU/nrm; V[t]=aV/nrm
            Qnl[t]=aQnl/nrm; Unl[t]=aUnl/nrm; Vnl[t]=aVnl/nrm
        end
    end
    return nothing
end

function stokes_timeseries_channelised(zR, zL, s, inv_gamma::Float64=1.0; rawcoh=nothing)
    nsamp = length(zR)
    nblk  = nsamp ÷ M_chan
    ntime = nblk ÷ N_ave
    I = zeros(Float64, ntime); Q = zeros(Float64, ntime)
    U = zeros(Float64, ntime); V = zeros(Float64, ntime)
    Qnl = zeros(Float64, ntime); Unl = zeros(Float64, ntime)
    Vnl = zeros(Float64, ntime)
    ph = exp(+im * phase_offset_RL[s])
    gR = gain_R[s]; gL = gain_L[s]
    ibpR = bandpass_R === nothing ? nothing : 1.0 ./ max.(bandpass_R[s], 1e-12)
    ibpL = bandpass_L === nothing ? nothing : 1.0 ./ max.(bandpass_L[s], 1e-12)
    dR_v = d_R[s]; dL_v = d_L[s]
    do_leak = any(!iszero, dR_v) || any(!iszero, dL_v)
    mQV_v = mQV[s]; mUV_v = mUV[s]
    do_vleak = any(!iszero, mQV_v) || any(!iszero, mUV_v)
    fch_MHz = [fmin_sub[s] + c*(fs/M_chan/1e6) for c = 1:NCHAN]
    # residual R/L delay (sub-ns, |tau| dnu < 1e-3) as a per-channel phase on L, after the stage-1 ramp
    delayph = tau_resid_RL[s] != 0.0 ? ComplexF64[cis(2π * fch_MHz[c]*1e6 * tau_resid_RL[s]) for c = 1:NCHAN] : nothing
    rawcoh === nothing || fill!(rawcoh, 0.0)
    if RM != 0.0
        λ = C_MPS ./ (fch_MHz .* 1e6); φ = 2 .* RM .* λ.^2
        fcs = cos.(φ); fsn = sin.(φ)
    else
        fcs = ones(NCHAN); fsn = zeros(NCHAN)
    end
    E = Float64[]; EC = ComplexF64[]
    _stokes_kernel!(I, Q, U, V, Qnl, Unl, Vnl, zR, zL, rawcoh === nothing ? zeros(0,0,0) : rawcoh, rawcoh !== nothing,
                    ibpR === nothing ? E : ibpR, ibpL === nothing ? E : ibpL, ibpR !== nothing, ibpL !== nothing,
                    Float64(gR), Float64(gL), inv_gamma, ComplexF64(ph), delayph === nothing ? EC : delayph, delayph !== nothing, Vector{Float64}(chan_w[:, s]),
                    dR_v, dL_v, do_leak, mQV_v, mUV_v, do_vleak, Vector{Float64}(fcs), Vector{Float64}(fsn),
                    Int(M_chan), Int(N_ave), ntime)
    return I, Q, U, V, Qnl, Unl, Vnl, ntime
end

const PLAN_CACHE = Dict{Int,Tuple{Any,Any}}()
function plans_for(Lwin)
    # plan_fft / plan_ifft (ESTIMATE), built under a lock so threads can share them
    lock(_KLOCK) do
        haskey(PLAN_CACHE, Lwin) && return PLAN_CACHE[Lwin]
        p = (plan_fft(zeros(ComplexF64, Lwin)), plan_ifft(zeros(ComplexF64, Lwin)))
        PLAN_CACHE[Lwin] = p; return p
    end
end

"""Stokes profiles of one read window, kept span = the window minus the edge crop. rawcoh=true appends a 9th return
value, the uncalibrated coherency (4, NCHAN, n_subbands, ntime) = RR, LL, Re RL, Im RL per subband, channel, 1-us sample."""
function process_window(buf::Vector{UInt32}, inv_gamma::Float64=1.0; rawcoh::Bool=false)
    Lwin  = length(buf)
    ncrop = floor(Int, edge_crop_ms*1e-3*fs)
    cropped_len = Lwin - 2*ncrop
    cropped_len < TS_NAVG && error("window too short after edge crop")
    fplan, iplan = plans_for(Lwin)
    i0f = ncrop + 1
    return process_window_span(buf, inv_gamma, i0f, cropped_len, fplan, iplan; rawcoh=rawcoh)
end

"""process_window for an arbitrary kept span z[i0f : i0f+keep-1] of the dedispersed window (process_window keeps
Lwin - 2 ncrop from ncrop+1, the fold keeps 2 ms after a 0.1-ms margin). t_axis follows the catalogue window convention."""
function process_window_span(buf::AbstractVector{UInt32}, inv_gamma::Float64, i0f::Int, cropped_len::Int, fplan, iplan;
                             rawcoh::Bool=false)
    Lwin = length(buf)
    (i0f >= 1 && i0f + cropped_len - 1 <= Lwin) || error("kept span outside the window")
    RC = nothing
    Iprof=nothing; Qprof=nothing; Uprof=nothing; Vprof=nothing
    Qnlprof=nothing; Unlprof=nothing; Vnlprof=nothing   # pre-D-term nebular cross-hand + V (calib check)
    with_ws(Lwin) do ws
        for s = 1:n_subbands
            beta, betaL = cached_operators(Lwin, s)
            stage1_pair!(ws, buf, s, beta, betaL, fplan, iplan)
            zR = view(ws.zR, i0f : i0f+cropped_len-1)
            zL = view(ws.zL, i0f : i0f+cropped_len-1)
            nt_s = (cropped_len ÷ M_chan) ÷ N_ave
            rc_s = rawcoh ? zeros(Float64, 4, NCHAN, nt_s) : nothing
            Is, Qs, Us, Vs, Qnls, Unls, Vnls, ntime = stokes_timeseries_channelised(zR, zL, s, inv_gamma;
                                                           rawcoh=rc_s)
            if Iprof === nothing
                Iprof=zeros(ntime); Qprof=zeros(ntime); Uprof=zeros(ntime); Vprof=zeros(ntime)
                Qnlprof=zeros(ntime); Unlprof=zeros(ntime); Vnlprof=zeros(ntime)
                rawcoh && (RC = zeros(Float64, 4, NCHAN, n_subbands, ntime))
            end
            rawcoh && (RC[:, :, s, :] .= rc_s)
            Iprof.+=Is; Qprof.+=Qs; Uprof.+=Us; Vprof.+=Vs
            Qnlprof.+=Qnls; Unlprof.+=Unls; Vnlprof.+=Vnls
        end
    end
    δt = TS_NAVG/fs
    t0 = t_start_s + edge_crop_ms*1e-3
    t_axis = t0 .+ ((1:length(Iprof)) .- 0.5) .* δt
    rawcoh && return Iprof, Qprof, Uprof, Vprof, t_axis, Qnlprof, Unlprof, Vnlprof, RC
    return Iprof, Qprof, Uprof, Vprof, t_axis, Qnlprof, Unlprof, Vnlprof
end

# Boxcar matched-filter search on the baseline-subtracted I profile.
# For each width W in BOXCAR_WIDTHS, slide a running sum of (I-μI) and score
# SNR = Σ(I-μI) / (√W · σI) (independent bins → √W noise scaling). Returns the
# (width, start-index, SNR) that maximise the score over all widths/positions.
# W=1 reduces exactly to the previous single-bin SNR (I[ipk]-μI)/σI.
function boxcar_search(Iprof, μI, σI)
    n = length(Iprof)
    bestsnr = -Inf; bestW = 1; besti0 = 1
    invσ = 1.0 / max(σI, 1e-12)
    @inbounds for W in BOXCAR_WIDTHS
        W > n && continue
        nrm = invσ / sqrt(W)
        s = 0.0
        for j = 1:W; s += Iprof[j]; end
        snr = (s - W*μI) * nrm
        if snr > bestsnr; bestsnr = snr; bestW = W; besti0 = 1; end
        for i = 2:(n - W + 1)
            s += Iprof[i+W-1] - Iprof[i-1]
            snr = (s - W*μI) * nrm
            if snr > bestsnr; bestsnr = snr; bestW = W; besti0 = i; end
        end
    end
    return bestW, besti0, bestsnr
end

wrap90(x) = mod(x + 90.0, 180.0) - 90.0
circmean_pa(v) = rad2deg(atan(sum(sind, 2 .* v), sum(cosd, 2 .* v))) / 2
unwrap_to(x, ref) = x - 180.0*round((x - ref)/180.0)

function parallactic_deg(utc::DateTime)
    jd   = datetime2julian(utc)
    d    = jd - 2451545.0
    T    = d / 36525.0
    gmst = 280.46061837 + 360.98564736629*d + 0.000387933*T^2 - T^3/38710000.0
    lst  = mod(gmst + USUDA_LON_DEG, 360.0)
    H    = deg2rad(mod(lst - CRAB_RA_DEG + 180.0, 360.0) - 180.0)
    φ    = deg2rad(USUDA_LAT_DEG); δ = deg2rad(CRAB_DEC_DEG)
    return rad2deg(atan(sin(H), tan(φ)*cos(δ) - sin(δ)*cos(H)))
end

# Hour angle [rad] for the Crab at Usuda (same reduction as parallactic_deg above).
@inline function _hour_angle_rad(utc::DateTime)
    jd   = datetime2julian(utc)
    d    = jd - 2451545.0
    T    = d / 36525.0
    gmst = 280.46061837 + 360.98564736629*d + 0.000387933*T^2 - T^3/38710000.0
    lst  = mod(gmst + USUDA_LON_DEG, 360.0)
    return deg2rad(mod(lst - CRAB_RA_DEG + 180.0, 360.0) - 180.0)
end

function elevation_deg(utc::DateTime)
    H = _hour_angle_rad(utc); φ = deg2rad(USUDA_LAT_DEG); δ = deg2rad(CRAB_DEC_DEG)
    return rad2deg(asin(sin(φ)*sin(δ) + cos(φ)*cos(δ)*cos(H)))
end

function azimuth_deg(utc::DateTime)
    H = _hour_angle_rad(utc); φ = deg2rad(USUDA_LAT_DEG); δ = deg2rad(CRAB_DEC_DEG)
    sin_el = sin(φ)*sin(δ) + cos(φ)*cos(δ)*cos(H)
    cos_el = sqrt(max(1.0 - sin_el^2, 0.0))
    cos_el < 1e-12 && return 0.0
    sin_az = -cos(δ)*sin(H) / cos_el
    cos_az = (sin(δ) - sin(φ)*sin_el) / (cos(φ)*cos_el)
    return mod(rad2deg(atan(sin_az, cos_az)), 360.0)
end

# BWG feed angle chi(t) = q + alpha*El + gamma*Az -- the curve the calibrated
# off-pulse nebular PA should track if the calibration is applied correctly.
feed_angle_deg(utc::DateTime) =
    parallactic_deg(utc) + BWG_ALPHA*elevation_deg(utc) + BWG_GAMMA*azimuth_deg(utc)

function parse_float_safe(s::AbstractString)
    try
        return parse(Float64, s)
    catch
        m = match(r"^([+-]?\d*\.?\d+)[eE]([+-]?\d+)$", strip(s))
        if m !== nothing
            base = parse(Float64, m.captures[1])
            expo = parse(Int, m.captures[2])
            return base * 10.0^expo
        end
        rethrow()
    end
end

# Detection-list loader. The RHCP and LHCP lists are merged by rotation number Nseq, keeping the larger list S/N of the
# two hands, and an entry is kept if that S/N reaches MIN_DET_SNR. The observing date is read from the file name.
# Returns Dict(nseq => (sod, phase, detsnr)), obsdate.
function load_detlist(files::Vector{String})
    d = Dict{Int,NamedTuple{(:sod,:phase,:detsnr),Tuple{Float64,Float64,Float64}}}(); obsdate = nothing
    files = sort(files)
    for f in files; isfile(f) || error("detection list not found: $f"); end
    for f in files
        if obsdate === nothing
            m = match(r"(\d{8})", basename(f))
            m !== nothing && (s = m.captures[1];
                obsdate = Date(parse(Int,s[1:4]), parse(Int,s[5:6]), parse(Int,s[7:8])))
        end
        for ln in eachline(f)
            (isempty(ln) || startswith(strip(ln), "#")) && continue
            t = split(ln); length(t) < 6 && continue
            nseq   = round(Int, parse_float_safe(t[4]))
            sod    = parse_float_safe(t[2])
            phase  = parse_float_safe(t[5])
            detsnr = parse_float_safe(t[6])
            detsnr < MIN_DET_SNR && continue
            # keep the entry with the higher list S/N of the two hands
            if !haskey(d, nseq) || detsnr > d[nseq].detsnr
                d[nseq] = (sod=sod, phase=phase, detsnr=detsnr)
            end
        end
    end
    @printf("detlist   : %d entries (S/N >= %.1f) from %d file(s); obs date %s\n",
            length(d), MIN_DET_SNR, length(files),
            obsdate === nothing ? "unknown" : string(obsdate))
    return d, obsdate
end

function pol_fractions(Iv, Qv, Uv, Vv, sQU, sV, sI)
    L2 = Qv^2 + Uv^2
    Lp = DEBIAS_POL ? sqrt(max(0.0, L2 - sQU^2))               : sqrt(L2)
    Pp = DEBIAS_POL ? sqrt(max(0.0, L2 + Vv^2 - sQU^2 - sV^2))  : sqrt(L2 + Vv^2)
    iI = Iv != 0 ? 1.0/Iv : NaN
    LIp = isnan(iI) ? NaN : min(100*Lp*iI, 100.0)
    VIp = isnan(iI) ? NaN : clamp(100*Vv*iI, -100.0, 100.0)
    PIp = isnan(iI) ? NaN : min(100*Pp*iI, 100.0)
    Pd  = max(Pp, sqrt(sQU^2 + sV^2))
    LIe = isnan(iI) ? NaN : 100*abs(iI)*sqrt(sQU^2 + (Lp*iI)^2*sI^2)
    VIe = isnan(iI) ? NaN : 100*abs(iI)*sqrt(sV^2  + (Vv*iI)^2*sI^2)
    PIe = isnan(iI) ? NaN : 100*abs(iI)*sqrt((L2*sQU^2 + Vv^2*sV^2)/Pd^2 + (Pp*iI)^2*sI^2)
    return (LIp, VIp, PIp, LIe, VIe, PIe, Lp, Pp)
end

# Write the HDF5 population archive. /catalog = per-pulse scalars under a documented
# DEFAULT reduction (bgsub on, Wardle-Kronberg debiased, 3σ aperture). /raw_profiles =
# calibrated Stokes, NOT baseline-subtracted, NOT debiased (full reversibility).
# /metadata = observation-level attributes. The τ-resampled /profiles cube is left to
# Python (a reversible op on /raw_profiles), per the archive design.
function write_archive(rows, profs, tag, outdir)
    isempty(profs) && (@printf("no profiles to archive\n"); return)
    Np = length(profs); Ns = length(profs[1].I)
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

        # ---- /metadata ----
        md = create_group(f, "metadata"); A = attributes(md)
        A["source_name"] = "Crab Pulsar"; A["telescope"] = "Usuda 64m"; A["backend"] = "S-band 2-pol"
        A["center_frequency_MHz"] = 2258.0; A["bandwidth_MHz"] = 128.0
        A["time_resolution_us"] = TS_NAVG / fs * 1e6
        A["polarization_calibrated"] = "bandpass+gain+diffgain(t)+R/L phase+delay+D-term(kappa)+ellipticity(Q,U->V)"
        A["provenance"] = provenance_string()
        A["DM_pc_cm3"] = DM
        A["delay_method"] = "R/L delay as a sky-frequency phase ramp on L before channelisation"
        A["parallactic_corrected"] = "catalog PA_* are in the feed frame; sky PA = wrap90(PA - nebula_diag/chi_BWG_deg)"
        A["debias_method"] = "Wardle-Kronberg / Everett-Weisberg 2001 (catalog only; raw_profiles undebiased)"
        A["normalization_method"] = "catalog: first-20% baseline subtracted; raw_profiles: none"
        A["software_version"] = "usuda-polarimetry"; A["creation_date"] = string(Dates.now())
    end
    @printf("hdf5 -> %s   (%d pulses, %d-sample profiles)\n", path, Np, Ns)
end

# Union aperture on the calibrated band-summed I profile, with the window's own mu, sigma (bins 1:nb). Candidates are
# every maximal run of bins with (I - mu) > 3 sigma, other than the main run [ilo, ihi], whose maximum reaches
# UNION_PK_NSIG sigma and lies within UNION_WMAX_US of the main peak. The union is [min start, max end] of the main run
# and all candidates, gaps included, so a multi-component pulse is measured as one pulse. flag = number of runs merged.
const UNION_PK_NSIG = Float64(cfgget(CFG, "aperture", "union_peak_nsig", 5.0))   # a 5-sigma peak at 1 us is not a noise run
const UNION_WMAX_US = Float64(cfgget(CFG, "aperture", "union_radius_us", 30.0))   # us
function union_aperture(Iprof, μI, σI, ilo::Int, ihi::Int)
    thr = ONPULSE_NSIG*σI; thr5 = UNION_PK_NSIG*σI
    wmax = round(Int, UNION_WMAX_US / ((TS_NAVG/fs)*1e6))
    ipk_run = ilo - 1 + argmax(@view Iprof[ilo:ihi])
    a = ilo; b = ihi; nmerge = 0; n = length(Iprof); i = 1
    while i <= n
        if (Iprof[i]-μI) > thr
            j = i
            while j < n && (Iprof[j+1]-μI) > thr; j += 1; end
            if !(i == ilo && j == ihi)
                k = i - 1 + argmax(@view Iprof[i:j])
                if (Iprof[k]-μI) >= thr5 && abs(k - ipk_run) <= wmax
                    nmerge += 1; a = min(a, i); b = max(b, j)
                end
            end
            i = j + 1
        else
            i += 1
        end
    end
    return a, b, nmerge, ipk_run
end

"""Per-window measurement from band-summed profiles. stats replaces the first-20 % statistics (nb, mu_I, sigma_I, mu_Q,
mu_U, mu_V, sigma_QU, sigma_V) and nebfn the off-pulse means ((ilo, ihi, offmask) -> I, Q, U, V, Q_preD, U_preD, V_preD)
when the caller computes them from another calibration. Returns (prof, row)."""
function measure_from_profiles(Iprof, Qprof, Uprof, Vprof, t, Qnlprof, Unlprof, Vnlprof,
                               nseq, sod, phase, detsnr, obsdate; stats=nothing, nebfn=nothing)
    if stats === nothing
        nb = max(10, floor(Int, BASELINE_FRAC*length(Iprof)))
        μI = mean(@view Iprof[1:nb]); σI = std(@view Iprof[1:nb])
        μQ = mean(@view Qprof[1:nb]); μU = mean(@view Uprof[1:nb]); μV = mean(@view Vprof[1:nb])
        σQU = 0.5*(std(@view Qprof[1:nb]) + std(@view Uprof[1:nb]))
        σV  = std(@view Vprof[1:nb])
    else
        nb, μI, σI, μQ, μU, μV, σQU, σV = stats
    end
    bw, i0, snr = boxcar_search(Iprof, μI, σI)
    i1  = i0 + bw - 1
    tpk = 0.5*(t[i0] + t[i1]); valid = PK_LO <= tpk <= PK_HI
    box_us = bw * (TS_NAVG/fs) * 1e6

    ipk = i0 - 1 + argmax(@view Iprof[i0:i1])
    b1  = sqrt(1.0 + 1.0/nb)
    sQU1, sV1, sI1 = σQU*b1, σV*b1, σI*b1
    Ipk_=Iprof[ipk]-μI; Qpk_=Qprof[ipk]-μQ; Upk_=Uprof[ipk]-μU; Vpk_=Vprof[ipk]-μV
    (LIpk,VIpk,PIpk,LIpk_e,VIpk_e,PIpk_e,Lpk,Ppk) = pol_fractions(Ipk_,Qpk_,Upk_,Vpk_,sQU1,sV1,sI1)
    PApk   = rad2deg(0.5*atan(Upk_,Qpk_))
    PApk_e = rad2deg(0.5*sQU1/max(hypot(Qpk_,Upk_),1e-12))

    thr = ONPULSE_NSIG*σI
    ilo = ipk; while ilo>1 && (Iprof[ilo-1]-μI) > thr; ilo -= 1; end
    ihi = ipk; while ihi<length(Iprof) && (Iprof[ihi+1]-μI) > thr; ihi += 1; end
    Won = ihi-ilo+1; bO = sqrt(1.0 + Won/nb)
    sQUo, sVo, sIo = sqrt(Won)*σQU*bO, sqrt(Won)*σV*bO, sqrt(Won)*σI*bO
    Iin=sum(@view Iprof[ilo:ihi])-Won*μI; Qin=sum(@view Qprof[ilo:ihi])-Won*μQ
    Uin=sum(@view Uprof[ilo:ihi])-Won*μU; Vin=sum(@view Vprof[ilo:ihi])-Won*μV
    (LIin,VIin,PIin,LIin_e,VIin_e,PIin_e,Lin,Pin) = pol_fractions(Iin,Qin,Uin,Vin,sQUo,sVo,sIo)
    PAin   = rad2deg(0.5*atan(Uin,Qin))
    PAin_e = rad2deg(0.5*sQUo/max(hypot(Qin,Uin),1e-12))
    Won_us = Won*(TS_NAVG/fs)*1e6
    Weq_us = (Ipk_!=0 ? Iin/Ipk_ : NaN)*(TS_NAVG/fs)*1e6

    th = Float64[atan(Uprof[j]-μU, Qprof[j]-μQ) for j in ilo:ihi]
    lw = Float64[hypot(Qprof[j]-μQ, Uprof[j]-μU) for j in ilo:ihi]
    Rb = abs(sum(lw .* cis.(th)) / max(sum(lw),1e-12))
    PA_std = rad2deg(sqrt(max(-2*log(max(Rb,1e-9)),0.0)))/2
    PA_rng = length(th)<2 ? 0.0 : rad2deg(maximum(th)-minimum(th))/2
    Lsnr_pk = hypot(Qpk_,Upk_)/max(sQU1,1e-12)
    Vsnr_in = abs(Vin)/max(sVo,1e-12)

    # union aperture: the same expressions as the 3-sigma block above, on [ulo, uhi]
    ulo, uhi, unflag, ipk_run = union_aperture(Iprof, μI, σI, ilo, ihi)
    Wu = uhi-ulo+1; bOu = sqrt(1.0 + Wu/nb)
    sQUu, sVu, sIu = sqrt(Wu)*σQU*bOu, sqrt(Wu)*σV*bOu, sqrt(Wu)*σI*bOu
    Iun=sum(@view Iprof[ulo:uhi])-Wu*μI; Qun=sum(@view Qprof[ulo:uhi])-Wu*μQ
    Uun=sum(@view Uprof[ulo:uhi])-Wu*μU; Vun=sum(@view Vprof[ulo:uhi])-Wu*μV
    (LIun,VIun,PIun,LIun_e,VIun_e,PIun_e,Lun,Pun) = pol_fractions(Iun,Qun,Uun,Vun,sQUu,sVu,sIu)
    PAun   = rad2deg(0.5*atan(Uun,Qun))
    PAun_e = rad2deg(0.5*sQUu/max(hypot(Qun,Uun),1e-12))
    Wun_us = Wu*(TS_NAVG/fs)*1e6

    utc = obsdate === nothing ? nothing :
          DateTime(obsdate) + Millisecond(round(Int, sod*1000))

    # --- off-pulse nebula diagnostic (per-window calibration monitor) ---
    # The Crab Nebula fills the beam and dominates every off-pulse bin, so the window-averaged off-pulse (Q, U) is the
    # calibrated nebular linear polarization at this window's feed angle. With a correct calibration PA_neb tracks the
    # feed angle chi(t), and a phase or sign error shows up as a collapsed |L| and a PA that no longer follows chi.
    offmask = trues(length(Iprof))
    gpad = max(3, ihi - ilo + 1)          # blank the pulse +/- one pulse-width
    offmask[max(1, ilo-gpad):min(length(Iprof), ihi+gpad)] .= false
    # Full chain (D-terms and ellipticity applied), the same transform the stored pulse profiles receive.
    if nebfn === nothing
    Ineb = mean(Iprof[offmask])
    Qneb = mean(Qprof[offmask]); Uneb = mean(Uprof[offmask]); Vneb = mean(Vprof[offmask])
    QnebP = mean(Qnlprof[offmask]); UnebP = mean(Unlprof[offmask]); VnebP = mean(Vnlprof[offmask])
    else
        Ineb, Qneb, Uneb, Vneb, QnebP, UnebP, VnebP = nebfn(ilo, ihi, offmask)
    end
    PA_neb = rad2deg(0.5*atan(Uneb, Qneb)); L_neb = hypot(Qneb, Uneb)
    # Before the leakage stage (delay, phase and RM only). V_neb_preD still carries the (Q,U) -> V leak, so V_neb_preD
    # against V_neb shows the ellipticity correction.
    PA_neb_preD = rad2deg(0.5*atan(UnebP, QnebP)); L_neb_preD = hypot(QnebP, UnebP)
    if utc !== nothing
        q_geo = parallactic_deg(utc); el_geo = elevation_deg(utc); az_geo = azimuth_deg(utc)
        chi_bwg = q_geo + BWG_ALPHA*el_geo + BWG_GAMMA*az_geo
    else
        q_geo = NaN; el_geo = NaN; az_geo = NaN; chi_bwg = NaN
    end

    # --- crop the stored raw profile to a window around the peak ---
    # Everything above this line (μI, σI, Won, ilo:ihi, all catalog quantities)
    # was already computed from the FULL Iprof/Qprof/Uprof/Vprof, so this crop
    # changes nothing in /catalog -- it only shrinks what /raw_profiles stores.
    win_len = 2*RAW_HALFWIN + 1
    Ifix = fill(NaN, win_len); Qfix = fill(NaN, win_len)
    Ufix = fill(NaN, win_len); Vfix = fill(NaN, win_len)
    @inbounds for (fidx, j) in enumerate((ipk-RAW_HALFWIN):(ipk+RAW_HALFWIN))
        if 1 <= j <= length(Iprof)
            Ifix[fidx] = Iprof[j]; Qfix[fidx] = Qprof[j]
            Ufix[fidx] = Uprof[j]; Vfix[fidx] = Vprof[j]
        end
    end
    ipk_rel_crop = RAW_HALFWIN   # peak is always centered now, 0-based

    prof = ((I=Ifix, Q=Qfix, U=Ufix, V=Vfix,
                  ipk_rel=ipk_rel_crop,
                  mI=μI, sI=σI, sQU=σQU, sV=σV,
                  Ipk=Ipk_, Qpk=Qpk_, Upk=Upk_, Vpk=Vpk_, Lpk=Lpk, Ppk=Ppk,
                  LIpk=LIpk, VIpk=VIpk, PIpk=PIpk, LIpk_e=LIpk_e, VIpk_e=VIpk_e, PIpk_e=PIpk_e,
                  PApk=PApk, PApk_e=PApk_e,
                  Iin=Iin, Qin=Qin, Uin=Uin, Vin=Vin, Lin=Lin, Pin=Pin,
                  LIin=LIin, VIin=VIin, PIin=PIin, LIin_e=LIin_e, VIin_e=VIin_e, PIin_e=PIin_e,
                  PAin=PAin, PAin_e=PAin_e, Won_us=Won_us, Weq_us=Weq_us,
                  PA_std=PA_std, PA_rng=PA_rng, Lsnr_pk=Lsnr_pk, Vsnr_in=Vsnr_in,
                  Iun=Iun, Qun=Qun, Uun=Uun, Vun=Vun, Lun=Lun, Pun=Pun,
                  LIun=LIun, VIun=VIun, PIun=PIun, LIun_e=LIun_e, VIun_e=VIun_e, PIun_e=PIun_e,
                  PAun=PAun, PAun_e=PAun_e, sIun=sIu, sQUun=sQUu, sVun=sVu, Wun_us=Wun_us,
                  union_flag=unflag, peak_bin=ipk, ap3_lo=ilo, ap3_hi=ihi, union_lo=ulo, union_hi=uhi,
                  ipk_run=ipk_run))
    row = ((nseq=string(nseq), phase=phase, detsnr=detsnr, valid=valid,
                 snr=snr, boxw=bw, boxus=box_us, sod=sod, utc=utc,
                 q_deg=q_geo, el_deg=el_geo, az_deg=az_geo, chi_bwg=chi_bwg,
                 PA_neb=PA_neb, L_neb=L_neb, I_neb=Ineb, V_neb=Vneb,
                 PA_neb_preD=PA_neb_preD, L_neb_preD=L_neb_preD, V_neb_preD=VnebP, tpk=tpk))
    return prof, row
end

"Process one read window and measure it. rawcoh=true also returns the uncalibrated coherency sums (extra)."
function measure_window(buf::Vector{UInt32}, nseq, sod, phase, detsnr, obsdate; rawcoh::Bool=false)
    # differential-gain drift gamma at this pulse's time
    inv_gamma = 1.0 / diffgain_at(sod)
    pw = process_window(buf, inv_gamma; rawcoh=rawcoh)
    Iprof, Qprof, Uprof, Vprof, t, Qnlprof, Unlprof, Vnlprof = pw[1:8]

    prof, row = measure_from_profiles(Iprof, Qprof, Uprof, Vprof, t, Qnlprof, Unlprof, Vnlprof,
                                      nseq, sod, phase, detsnr, obsdate)
    nb = max(10, floor(Int, BASELINE_FRAC*length(Iprof)))
    ilo = prof.ap3_lo; ihi = prof.ap3_hi; ulo = prof.union_lo; uhi = prof.union_hi
    Won = ihi-ilo+1; Wu = uhi-ulo+1
    offmask = trues(length(Iprof))
    gpad = max(3, ihi - ilo + 1)
    offmask[max(1, ilo-gpad):min(length(Iprof), ihi+gpad)] .= false
    extra = nothing
    if rawcoh
        RC = pw[9]
        RCb  = dropdims(sum(@view(RC[:, :, :, 1:nb]), dims=4), dims=4) ./ nb
        RCon = dropdims(sum(@view(RC[:, :, :, ilo:ihi]), dims=4), dims=4) .- Won .* RCb
        RCof = dropdims(sum(@view(RC[:, :, :, offmask]), dims=4), dims=4) ./ count(offmask)
        RCun = dropdims(sum(@view(RC[:, :, :, ulo:uhi]), dims=4), dims=4) .- Wu .* RCb
        extra = (RC_on=RCon, RC_off=RCof, RC_base=RCb, ilo=ilo, ihi=ihi, nb=nb, offmask=offmask, profiles=pw,
                 RC_on_union=RCun, union_lo=ulo, union_hi=uhi)
    end
    return prof, row, extra
end
