using ASP

facts("as_nnls") do

    context("Wide and sparse NNLS problem.") do
        
        m, n = 600, 2560              # No. of rows, cols
        k = 20                        # No. of nonzeros
        p = randperm(n)[1:k]          # Position of nonzeros in x
        x = zeros(n)                  # Generate sparse nonnegative solution
        x[p] = max(randn(k),0)
        A = randn(m,n)                # Gaussian m-by-n ensemble
        b = A*x                       # Compute the RHS vector
        x, r, inform = as_nnls(A,b)   # Solve the nonnegative LS problem

        g = -(A'*r)
        @fact vecnorm( min(x,g) ) --> less_than(1e-8)
        
    end

    context("Overdetermined NNLS problem.") do

        m, n = 100, 10
        x = max(randn(n), 0)
        A = randn(m,n)
        b = A*x
        x, r, inform = as_nnls(A, b)
    
        g = -(A'*r)
        @fact vecnorm( min(x,g) ) --> less_than(1e-8)
    end

end


