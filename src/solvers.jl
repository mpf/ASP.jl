"""
Solve the nonnegative least-squares problem.

   [X,INFORM] = AS_NNLS(A,B) solves the problem

     minimize_x  1/2 ||Ax-B||_2^2        subject to  x >= 0.

   AS_NNLS(A,B,C) additionally includes a linear term in the objective:

     minimize_x  1/2 ||Ax-B||_2^2 + C'x  subject to  x >= 0.

   AS_NNLS(A,B,C,OPTS) specifies options that can be set using
   AS_SETPARMS.

   AS_NNLS(A,B,C,OPTS,INFORM) uses information stored in INFORM
   (from a previous call to AS_NNLS) to warm-start the
   algorithm. Note that the previous call to AS_NNLS must have been
   to a problem with the same A and C.

   In all cases, the INFORM output argument is optional, and contains
   statistics on the solution process.

   Inputs
   A       is an m-by-n matrix, explicit or an operator.
   B       is an m-vector.
   C       is an n-vector.
   OPTS    is an options structure created using AS_SETPARMS.
   INFORM  is an information structure from a previous call to AS_NNLS.

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
function as_nnls(A::AbstractMatrix, b::Vector, kwargs...)

    m, n = size(A)
    bl = -Inf*ones(n)
    bu =     zeros(n)
    λ = 1.0
    
    (xx, r, inform) = bpdual(A, b, λ, bl, bu; kwargs...)

    # BPdual's solution xx only contains nonzero elements.
    # Make it full length.
    x = zeros(n)
    x[inform.active] = xx

    return (x, r, inform)
end # function as_nnls
