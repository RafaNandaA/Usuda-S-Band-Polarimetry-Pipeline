# Mean-pulse fold geometry. Includes the per-pulse kernel (kernel.jl) and adds the window layout used for the fold.
#
# Each folded rotation is read as three 7.5-ms windows (main pulse, off-pulse, low-frequency interpulse), each starting
# 1.1 ms before its component. A window gets its own dedispersion FFT and keeps 2 ms after a 0.1-ms margin (2000 samples of
# 1 us). The 5.4 ms after the kept span exceed the dispersion sweep over the band, so the kept span is fully dedispersed.

include(joinpath(@__DIR__, "kernel.jl"))

const FOLD_WREAD = round(Int, 7.5e-3*fs)     # words read per component window (480 000)
const FOLD_PRE  = round(Int, 0.1e-3*fs)     # samples before the kept span (6 400)
const FOLD_KEEP  = round(Int, 2.0e-3*fs)     # kept samples (128 000 = 2000 x 1 us)

"One fold window: returns I, Q, U, V, Qnl, Unl, Vnl over the kept span (calibrated with the read-time tables)."
function fold_window_span(buf::AbstractVector{UInt32}, inv_gamma::Float64=1.0; pre::Int=FOLD_PRE, keep::Int=FOLD_KEEP)
    fplan, iplan = plans_for(length(buf))
    r = process_window_span(buf, inv_gamma, pre+1, keep, fplan, iplan)
    return r[1], r[2], r[3], r[4], r[6], r[7], r[8]
end
