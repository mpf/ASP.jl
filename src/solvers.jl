"""
Solve the nonnegative least-squares problem.

   [X,INFORM] = AS_NNLS(A,B) solves the problem

     minimize_x  1/2 ||Ax-B||_2^2        subject to  x >= 0.

   AS_NNLS(A,B,C) additionally includes a linear term in the objective:

     minimize_x  1/2 ||Ax-B||_2^2 + C'x  subject to  x >= 0.

   AS_NNLS(A,B,C,OPTS) specifies options that can be set using
   AS_SETPARMS.

   The INFORM output argument contains statistics on the solution
   process.

   Inputs
   A       is an m-by-n matrix, explicit or an operator.
   B       is an m-vector.
   C       is an n-vector.
   OPTS    is an options structure created using AS_SETPARMS.

   Example
   m = 600; n = 2560; k = 20;    % No. of rows, columns, and nonzeros
   p = randperm(n); p = p(1:k);  % Position of nonzeros in x
   x = zeros(n,1);               % Generate sparse nonnegative solution
   x(p) = max(randn(k,1),0);
   A = randn(m,n);               % Gaussian m-by-n ensemble
   b = A*x;                      % Compute the RHS vector
   x = as_nnls(A,b);             % Solve the nonnegative LS problem

   See also `bpdual`.
"""
function as_nnls(A, b::Vector, kwargs...)

    m, n = size(A)
    bl = -Inf*ones(n)
    bu =     zeros(n)
    λ = 1.0
    
    xx, r, inform = bpdual(A, b, λ, bl, bu; kwargs...)

    # BPdual's solution xx only contains nonzero elements.
    x = xscatter(xx, n, inform.active)

    return (x, r, inform)
end # function as_nnls

"""
Solve a least-squares problem over the simplex. The call

    (x, inform) = as_simplex(A, b, τ)

solves the problem

     minimize  ½||Ax-b||^2  subj to  sum(x) = τ, x ≥ 0.

See also `as_nnls`, `bpdual`.
"""
function as_simplex(A, b::Vector, τ; kwargs...)

    m, n = size(A)
    bl = -Inf*ones(n)
    bu =     zeros(n)
    λ = 1.0
    λ₀ = 1e-5#√eps(1.0)
        
    function Aprod(x)
        z = Array{Float64}(m+1)
        z[1:m] = A*x
        z[m+1] = sum(x)/λ₀
        return z
    end

    function Atran(y)
        z = A'*y[1:m] + y[m+1]/λ₀
    end

    Abar = LinearOperator(m+1, n, Float64, false, false,
                          x->Aprod(x),
                          Nullable{Function}(),
                          y->Atran(y))
    bbar = [ b; τ/λ₀ ]

    xx, r, inform = bpdual(Abar, bbar, λ, bl, bu; kwargs...)

    x = xscatter(xx, n, inform.active) # scatter xx to full size x
    deleteat!(r, m+1)                  # remove artificial residual

    # check kkt conditions.
    μ = dot(x,A'*r) / τ # multiplier
    z = A'*r - μ      # reduced costs
    
    @printf(" %-20s: %8.1e\n","Pr infeasibility",
            max(sum(x) - τ, 0))
    @printf(" %-20s: %8.1e\n","Du infeasibility",
            norm(min(x,z),Inf)/max(norm(x),norm(z),1))
    @printf("\n")
    
    return (x, r, inform)
    
end

# Scatter short vector to full-length vector.
function xscatter(x, n, active)
    z = zeros(n)
    z[active] = x
    return z
end
