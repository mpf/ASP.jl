using ASP
using LinearOperators
#include("../src/BPdual.jl")

# Setup a simple test problem.
srand(0)
const m = 600
const n = 2560 
const k = 30                # No. of rows, columns, and nonzeros
p = randperm(n)[1:k]  # Position of nonzeros in x
x    = zeros(n)             # Generate sparse solution
x[p] = randn(k)
A = randn(m,n)     # Gaussian m-by-n ensemble
b = A*x            # Compute the RHS vector

#write_matfile("small.mat", A=A, b=b, lambda=λ)
const λ = 0.#0.1*norm(A'*b, Inf)

active = zeros(Int, 0)
state = zeros(Int, n)
S = zeros(m, 0)
R = zeros(0, 0)
y = zeros(m)
bl = -ones(n)
bu = +ones(n)

x, r, inform = bpdual(A, b, λ, bl, bu,
                      trim = 1, loglevel=1)





function partialDCTAdjoint(idx,n,y)
    z = zeros(n)
    z[idx] = y
    idct!(z)
    z *= sqrt(n)
    return z
end
function partialDCTForward(idx,n,x)
    z = dct(x) / sqrt(n)
    return z[idx]
end

m, n = 6000, 25600
bl = -ones(n)
bu = +ones(n)
k = 300
p = randperm(n)[1:k]
x = zeros(n)
x[p] = randn(k)
idx = randperm(n)[1:m]
A = LinearOperator(m, n, Float64, false, false,
                   v -> partialDCTForward(idx,n,v),
                   Nullable{Function}(),
                   w -> partialDCTAdjoint(idx,n,w))
b = A*x

@profile x, r, inform = bpdual(A, b, 0.0, bl, bu)

