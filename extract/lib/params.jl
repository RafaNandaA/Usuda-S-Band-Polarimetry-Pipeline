# Instrument, observation and processing parameters, and the calibration tables applied while reading.
# Values come from the configuration file named by ENV["USUDAPOL_CONFIG"]. The globals are not declared const where a
# table set can replace them, so a table set can be loaded into a running module.

include(joinpath(@__DIR__, "config.jl"))
const CFG = config_from_env()

# ------------------------------------------------------------------------------------------------ instrument
# Raw file: little-endian UInt32 words, each packing 8 simultaneous 4-bit offset-binary samples (value - 7.5),
# nibbles CH0-3 = right-circular (R) subbands 0-3 and CH4-7 = left-circular (L) subbands 0-3.
fs         = round(Int, cfgget(CFG, "observation", "sample_rate_MHz", 64.0) * 1e6)   # real sampling rate [Hz]
n_subbands = 4                                                                          # R+L subband pairs
fmin_sub   = Float64.(cfgget(CFG, "observation", "subband_lower_MHz", [2194.0, 2226.0, 2258.0, 2290.0]))   # lower edges [MHz], fs/2 = 32 MHz each
f_ref_MHz  = Float64(cfgget(CFG, "dedispersion", "ref_freq_MHz", 2322.0))               # top of the band, so the four subbands share one arrival time
DM         = Float64(cfgget(CFG, "dedispersion", "dm", 56.7094))                        # pc cm^-3

# ------------------------------------------------------------------------------------------------ read geometry
# Each listed pulse is read as one window of read_ms. Coherent dedispersion runs as one long FFT over the whole window,
# so the window must exceed the dispersion sweep (about 5.25 ms over the band) plus the crop at both edges.
t_start_s    = 0.055     # window start relative to the list-time reference of the read (s)
edge_crop_ms = 6.0       # removed at each edge after dedispersion, where the circular FFT wraps

# ------------------------------------------------------------------------------------------------ channelisation
# 64-point FFT of the dedispersed analytic signal: 32 channels of 1 MHz per subband, and with N_ave = 1 a 1-us sample.
M_chan = Int(cfgget(CFG, "channeliser", "m_chan", 64))
N_ave  = Int(cfgget(CFG, "channeliser", "n_ave", 1))

# ------------------------------------------------------------------------------------------------ R/L delay (read time)
# The R/L signal-path delay (about -91 ns, close to 6 samples) is applied as a sky-frequency phase ramp
# cis(2 pi f_sky tau_s) on L inside the long dedispersion FFT, before channelisation. The same delay applied after the
# 64-point FFT decorrelates the cross-hand and scales L, Q and U by about 0.92. A sub-ns residual fitted later from the
# nebula is applied per channel (tau_resid_RL below).
delay_RL = Float64.(cfgget(CFG, "delay", "rl_delay_s", [-9.03e-8, -9.064e-8, -9.164e-8, -9.206e-8]))

"Phase ramp for L's dedispersion operator: r[k+1] = cis(2 pi f_sky tau) at long-FFT bin k+1 (k = 1..N/2), DC untouched."
function ramp_L(Lwin::Int, f_lo::Float64, tau::Float64)
    r = ones(ComplexF64, Lwin); Nnyq = Lwin ÷ 2; df = fs / (Lwin * 1.0e6)
    @inbounds for k = 1:Nnyq
        r[k+1] = cis(2π * (f_lo + k*df) * 1e6 * tau)
    end
    r
end

# ------------------------------------------------------------------------------------------------ read-time tables
# Applied while reading only to the calibrated reference rows stored beside the uncalibrated coherency. The calibrated
# products are made later by calibrate/apply.jl from the stored coherency, with any table set. Order of the chain:
# bandpass -> gain -> gamma(t) -> R/L phase -> residual delay -> D-terms -> Stokes -> ellipticity -> RM.
# With no table set every term is the identity.
bandpass_R = nothing; bandpass_L = nothing          # per-channel amplitude per hand (nebula auto-power normalised)
gain_R = ones(n_subbands); gain_L = ones(n_subbands)
diffgain_sod = Float64[]; diffgain_gamma = Float64[]  # gamma(t): L/R gain drift, applied as L -> L / gamma(sod)
phase_offset_RL = zeros(n_subbands)                   # R/L phase per subband [rad], applied as L *= cis(phi_s)
d_R = [zeros(ComplexF64, M_chan ÷ 2) for _ in 1:n_subbands]   # leakage, r = r0 - d_R l, l = l0 - d_L r0
d_L = [zeros(ComplexF64, M_chan ÷ 2) for _ in 1:n_subbands]
mQV = [zeros(Float64, M_chan ÷ 2) for _ in 1:n_subbands]      # ellipticity, V -> V - m_QV Q - m_UV U (feed frame)
mUV = [zeros(Float64, M_chan ÷ 2) for _ in 1:n_subbands]
RM = 0.0                                              # rotation measure [rad m^-2], not measurable on this track (degenerate with tau_s, phi_s)
tau_resid_RL = zeros(n_subbands)                      # residual R/L delay [s] as a per-channel phase on L
chan_w = ones(Float64, M_chan ÷ 2, n_subbands)        # band-sum channel weights, 0 for channels left out of the band sums

"Directory of the read-time table set: [paths] readtime_tables_dir, else tables_dir if it exists, else none."
function readtime_tables_dir(cfg=CFG)
    P = get(cfg, "paths", Dict{String,Any}())
    d = get(P, "readtime_tables_dir", "")
    isempty(d) || return d
    t = get(P, "tables_dir", "")
    (!isempty(t) && isdir(t)) ? t : ""
end

"channel weights from dropped (subband, channel) pairs: n_ch/n_kept for kept channels of a subband, 0 for dropped"
function channel_weights(drop::Vector{Tuple{Int,Int}}, nch::Int=M_chan ÷ 2)
    W = ones(nch, n_subbands)
    for s in 1:n_subbands
        dr = [c for (ss, c) in drop if ss == s]; isempty(dr) && continue
        for c in 1:nch; W[c, s] = c in dr ? 0.0 : nch / (nch - length(dr)); end
    end
    W
end

"Load a table set (the four CSV tables and scalars.txt written by calibrate/fit.py) into this module's globals."
function load_readtime_tables!(dir::AbstractString)
    global bandpass_R, bandpass_L, gain_R, gain_L, phase_offset_RL, RM, tau_resid_RL, chan_w
    rd(f) = [split(l, ",") for l in eachline(joinpath(dir, f)) if !(isempty(l) || startswith(l, "#") || startswith(l, "subband") || startswith(l, "sod"))]
    if isfile(joinpath(dir, "gainbandpass_cal_M$(M_chan).csv"))
        bandpass_R = [Float64[] for _ = 1:n_subbands]; bandpass_L = [Float64[] for _ = 1:n_subbands]
        for p in rd("gainbandpass_cal_M$(M_chan).csv")
            s = parse(Int, p[1]); push!(bandpass_R[s], parse(Float64, p[4])); push!(bandpass_L[s], parse(Float64, p[5]))
        end
    end
    if isfile(joinpath(dir, "diffgain_time_M$(M_chan).csv"))
        empty!(diffgain_sod); empty!(diffgain_gamma)
        for p in rd("diffgain_time_M$(M_chan).csv"); push!(diffgain_sod, parse(Float64, p[1])); push!(diffgain_gamma, parse(Float64, p[2])); end
    end
    if isfile(joinpath(dir, "dterm_cal_python_M$(M_chan).csv"))
        for p in rd("dterm_cal_python_M$(M_chan).csv")
            s = parse(Int, p[1]); c = parse(Int, p[2])
            d_R[s][c] = complex(parse(Float64, p[4]), parse(Float64, p[5])); d_L[s][c] = complex(parse(Float64, p[6]), parse(Float64, p[7]))
        end
    end
    if isfile(joinpath(dir, "mueller_vleak_M$(M_chan).csv"))
        for p in rd("mueller_vleak_M$(M_chan).csv")
            s = parse(Int, p[1]); c = parse(Int, p[2]); mQV[s][c] = parse(Float64, p[3]); mUV[s][c] = parse(Float64, p[4])
        end
    end
    sf = joinpath(dir, "scalars.txt")
    if isfile(sf)
        for l in eachline(sf)
            (isempty(strip(l)) || startswith(l, "#")) && continue
            k, v = strip.(split(l, "=", limit=2))
            if k == "drop_channels"
                chan_w = v == "none" ? ones(M_chan ÷ 2, n_subbands) :
                    channel_weights([(parse(Int, split(q, ":")[1]), parse(Int, split(q, ":")[2])) for q in split(v, ",")])
                continue
            end
            x = parse.(Float64, split(v, ","))
            k == "phase_offset_RL" && (phase_offset_RL = x); k == "RM" && (RM = x[1])
            k == "gain_R" && (gain_R = x); k == "gain_L" && (gain_L = x); k == "tau_resid_RL" && (tau_resid_RL = x)
            # the stage-1 ramp cannot change after reading: a table set fitted with another ramp is refused
            k == "delay_RL" && (x == delay_RL || error("table set $dir has delay_RL $x, the read uses $delay_RL ([delay] rl_delay_s)"))
        end
    end
    dir
end

let d = readtime_tables_dir()
    isempty(d) || load_readtime_tables!(d)
end

"gamma(t) at a second of day: linear interpolation of the table, clamped at its ends (1 with no table)."
function diffgain_at(sod::Real)
    isempty(diffgain_sod) && return 1.0
    sod <= diffgain_sod[1]   && return diffgain_gamma[1]
    sod >= diffgain_sod[end] && return diffgain_gamma[end]
    i  = searchsortedlast(diffgain_sod, sod)
    t0 = diffgain_sod[i];   t1 = diffgain_sod[i+1]
    g0 = diffgain_gamma[i]; g1 = diffgain_gamma[i+1]
    return g0 + (g1 - g0) * (sod - t0) / (t1 - t0)
end

"One-line description of the reduction, written into every product."
provenance_string() = string("usuda-polarimetry extraction; DM = ", DM, " pc cm^-3 (reference ", f_ref_MHz, " MHz)",
    "; R/L delay applied as a sky-frequency phase ramp on L before channelisation, tau_s_ns = ",
    join(string.(round.(delay_RL .* 1e9; digits=3)), "/"),
    "; ", M_chan, "-point channeliser, N_ave = ", N_ave,
    "; read-time tables: ", (d = readtime_tables_dir(); isempty(d) ? "none (identity)" : basename(normpath(d))))
