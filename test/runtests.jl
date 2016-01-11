using ASP
using FactCheck

tests = ["as_nnls"]

for t in tests
    include("$(t).jl")
end
