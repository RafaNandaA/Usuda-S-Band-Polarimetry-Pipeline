# Configuration file reader shared by every Julia stage.
using TOML

"Read a TOML configuration file. Paths in [paths] are resolved relative to the directory of the file."
function load_config(path::AbstractString)
    cfg = TOML.parsefile(path)
    base = dirname(abspath(path))
    P = get!(cfg, "paths", Dict{String,Any}())
    for (k, v) in P
        if v isa String && !isempty(v) && !isabspath(v)
            P[k] = normpath(joinpath(base, v))
        end
    end
    cfg
end

"The configuration named by ENV[\"USUDAPOL_CONFIG\"] (set by every entry script before the kernels are loaded)."
function config_from_env()
    haskey(ENV, "USUDAPOL_CONFIG") || error("ENV[\"USUDAPOL_CONFIG\"] is not set: run the stage entry scripts with a configuration file")
    load_config(ENV["USUDAPOL_CONFIG"])
end

cfgget(cfg, sec, key, default) = get(get(cfg, sec, Dict{String,Any}()), key, default)
