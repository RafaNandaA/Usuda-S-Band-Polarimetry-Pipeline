# Gaussian self-noise sums per window from the calibrated dynamic spectra, for the analysis.
#
#   julia -t 4 analyze/selfnoise.jl CONFIG.toml OUT.tsv [nmax]
#
# A bright pulse adds its own noise to the integrated Stokes parameters (van Straten 2009, Eq. 28, one sample per 1-us cell
# and channel). For X in (I, Q, U, V) and each aperture (union, 3-sigma), over the kept channels with band weights
# a_c = w_c/32 (w_c = 32/n_kept in the subband):
#   A_X = sum_{c,t} a_c^2 (s2_Xc / mu_c^2) aX,   B_X = sum_{c,t} a_c^2 (s2_Xc / mu_c^2) aP
#   aP = P^2 - s2_Ic (1 + 2P/mu_c),  aX = X^2 - s2_Xc (1 + 2P/mu_c),  P = I_t - mu_c (baseline-subtracted cell), mu_c = baseline I
# so that sigma2_self,X(kappa) = A_X + c(kappa) B_X with c = 0.5 kappa/(2 - kappa/2), evaluated in the analysis.
# Reads the stored windows in paths.work_dir/items and the tables in paths.tables_dir (read-time tables: paths.readtime_tables_dir).
using Printf, Statistics, Base.Threads
length(ARGS) >= 2 || error("usage: julia -t 4 analyze/selfnoise.jl CONFIG.toml OUT.tsv [nmax]")
ENV["USUDAPOL_CONFIG"] = abspath(ARGS[1])
include(joinpath(@__DIR__, "..", "calibrate", "lib", "apply_lib.jl"))
const OUTF = ARGS[2]
const NMAX = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : typemax(Int)
const RUN = V2.CFG["paths"]["work_dir"]
const Trt = tables_from_module(V2, "readtime")
const Tp = load_tables(V2.CFG["paths"]["tables_dir"], Trt)
files = sort([f for f in readdir(joinpath(RUN, "items")) if startswith(f, "cat_") && endswith(f, ".h5")], by = f -> parse(Int, f[5:end-3]))
files = files[1:min(NMAX, length(files))]
keep = trues(128); keep[[32, 64, 96, 128, 33]] .= false          # channel 32 of each subband and channel 1 of subband 2
const A = zeros(128)
for s in 0:3
    r = s*32+1:s*32+32; nk = count(keep[r]); A[r] .= keep[r] .* (32 / nk) ./ 32
end
R = Vector{Vector{Float64}}(undef, length(files))
@threads :dynamic for i in eachindex(files)
    prof, row, freq, ex, dg = calibrate_window(joinpath(RUN, "items", files[i]), Tp, Trt)
    dyn, lo = ex.dyn, ex.dyn_lo
    out = Float64[]
    for (a, b) in (ex.union, ex.ap3)
        acc = zeros(8)
        for c in 1:128
            A[c] == 0 && continue
            mu = ex.fcal_base[1, c]
            s2 = [ex.sig_full[k, c]^2 for k in 1:4]
            for t in a-lo+1:b-lo+1
                P = dyn[1, c, t] - mu
                corr = 1 + 2P / mu
                aP = P^2 - s2[1] * corr
                for k in 1:4
                    X = dyn[k, c, t] - ex.fcal_base[k, c]
                    aX = X^2 - s2[k] * corr
                    w = A[c]^2 * s2[k] / mu^2
                    acc[k] += w * aX; acc[4+k] += w * aP
                end
            end
        end
        append!(out, acc); push!(out, b - a + 1)
    end
    R[i] = out
end
open(OUTF, "w") do io
    println(io, "cat\tAI_un\tAQ_un\tAU_un\tAV_un\tBI_un\tBQ_un\tBU_un\tBV_un\tW_un\tAI_3s\tAQ_3s\tAU_3s\tAV_3s\tBI_3s\tBQ_3s\tBU_3s\tBV_3s\tW_3s")
    for i in eachindex(files); println(io, files[i][5:end-3], "\t", join([@sprintf("%.10g", x) for x in R[i]], "\t")); end
end
println("wrote ", length(files), " windows")
