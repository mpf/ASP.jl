using ASP

facts("as_simplex") do

    context("solution x = (1/n,...,1/n)") do

        m, n = 100, 100                 # No. of rows, cols
        A = eye(m, n)                 # Gaussian m-by-n ensemble
        b = zeros(m)
        τ = 1.0
        x, r = ASP.as_simplex(A, b, τ)   # Solve the nonnegative LS problem
    
        @fact vecnorm( x-ones(n)/n ) --> less_than(1e-8)
    end

    context("random problem") do

        m, n = 1000, 20
        A = randn(m, n)
        x = max(0, randn(n))
        x = x / sum(x)
        τ = 1.0
        b = A*x

        x, r = ASP.as_simplex(A, b, τ)

        # check kkt conditions.
        μ = dot(x,A'*r) / τ # multiplier
        z = A'*r - μ      # reduced costs
        
        @fact abs(sum(x) - τ) --> less_than(1e-6)
        @fact norm( min(x,z) ) --> less_than(1e-6)
        
    end
    
end


