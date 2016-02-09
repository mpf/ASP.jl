using QRupdate
using LinearOperators

macro fprintf(args...)
    :( f()=@printf $(args...); f() )
end

type OptimResults
    # Statistics
    itns::Int
    time::Float64
    stat::Symbol
    exitmsg::AbstractString
    nprodA::Int
    nprodAt::Int
    numtrim::Int

    # State
    active::Vector{Int}
    state::Vector{Int}
    y::Vector{Float64}
    S::Matrix{Float64}
    R::Matrix{Float64}
end

# Constructor
function OptimResults(m::Int, n::Int)
    active = zeros(Int,0)
    state = zeros(Int,n)
    y = zeros(m)
    S = zeros(m,0)
    R = zeros(0,0)
    OptimResults(0,0,:exit_unknown,"Exit Unknown",0,0,0,
                 active, state, y, S, R)
end


"""
Solve the optimization problem

   DP:       minimize_y   - b'y  +  1/2 lambda y'y
             subject to   bl <= A'y <= bu
using given A,b,bl,bu,lambda.  When bl = -e, bu = e = ones(n,1),
DP is the dual Basis Pursuit problem

   BPdual:   maximize_y   b'y  -  1/2 lambda y'y
             subject to   ||A'y||_inf  <=  1.

General call to BPdual:
(active,state,x,y,S,R,info) = BPdual(A,b,bl,bu,lambda,active,state,y,S,R)

EXAMPLE
=======
Generate data:
[A,b,lambda] = gendata(1);

Solve BPdual:
[active,state,x,y,S,info] = BPdual(A,b,-1,1,lambda);

Resolve with a different lambda
lambda = lambda / 10;
[active,state,x,y,S,R,info] = BPdual(A,b,-1,1,lambda,active,state,y,S,R);

INPUT
=====
A          is an m-by-n linear operator.
b          is an m-vector.
lambda     is a nonnegative scalar.
bl, bu     are n-vectors.
active,state,y,S,R  may be empty (i.e., equal to [])
           or output from BPdual with some previous value of lambda.
           
OUTPUT
======
active     is an nact-vector of indices j with nact = length(active),
           listing which of the constraints bl(j) <= A(:,j)'y <= bu(j)
           is active.
state      is an n-vector giving the state of each dual constraint.
           state(j) = -2  if         Aj'*y < bl(j)  infeas below
                    = -1  if         Aj'*y = bl(j)  active below
                    = +1  if         Aj'*y = bu(j)  active above
                    = +2  if         Aj'*y > bu(j)  infeas above
                    =  0  if bl(j) < Aj'*y < bu(j)  inactive
x          is an nact-vector of primal solution values.
           The n-vector xx = zeros(n,1), xx(active) = x
           solves the primal BP problem (see below).
y          is an m-vector of BPdual solution values.
S          is the submatrix A(:,active).
R          is the Cholesky factorization of S.
info       is a structure with the following components:
           .time    total solution time (seconds)
           .stat    = 0  solution is optimal
                    = 1  too many iterations
                    = 2  current S is rank-deficient
                    = 3  dual infeasible point
           .itn     number of iterations
           .exitmsg exit message
           .nprodA  number of products with A
           .nprodAt number of products with A'
           .rNorm   2-norm of the final residual r = b - A(:,active)*x
           .xNorm   1-norm of the primal solution
           .numtrim number of constraints trimmed from final working set

BPdual Toolbox
Copyright 2008, Michael P. Friedlander and Michael A. Saunders
http://www.cs.ubc.ca/labs/scl/bpdual

The primal BP problem may be written

BP:        min_{x,y} ||x||_1 +  1/2 lambda||y||_2^2
              st         Ax     +      lambda  y  =  b.

If lambda > 0, we have the residual vector r = lambda y = b - Ax,
and BP is equivalent to the original BP-denoising problem

   BPDN:      min_{x,y} lambda ||x||_1 +  1/2 ||r||_2^2
              st         Ax     +      r  =  b.


09 Jun 2007: First version of BPdual.
             Michael Friedlander and Michael Saunders.
             A can be dense or sparse (but not an operator).
             S = A(:,active) holds active columns.
             Least-squares subproblems solved by x = S\g.
29 Aug 2007: "A" is now an operator.
09 Nov 2007: Generalize the l1 dual constraints from -e <= A'y <= e
             to bl <= A'y <= bu.
"""
function bpdual(# Positional arguments
                A::Union{AbstractMatrix,AbstractLinearOperator},
                b::Vector,
                λin::Real,
                bl::Vector,
                bu::Vector;

                # Keyword arguments
                inform::OptimResults = OptimResults(size(A)...),                
                coldstart::Bool = true,
                trim::Int = 0,
                itnMax::Int = 10*maximum(size(A)),
                feaTol::Real = 5e-05,
                λmin::Real = √eps(1.),
                optTol::Real = 1e-05,
                gapTol::Real = 1e-06,
                pivTol::Real = 1e-12,
                actMax::Real = Inf,
                loglevel::Int = 1)
    
    time0 = time()
    
    # ------------------------------------------------------------------
    # Check input arguments.
    # ------------------------------------------------------------------
    m, n = size(A)

    active = inform.active
    state = inform.state
    y = inform.y
    S = inform.S
    R = inform.R
    
    coldstart = isempty(S)

    # ------------------------------------------------------------------
    # Grab input options and set defaults where needed. 
    # ------------------------------------------------------------------
    tieTol = feaTol + 1e-8        # Perturbation to break ties
    λ = max(λin, λmin)

    # ------------------------------------------------------------------
    # Print log header.
    # ------------------------------------------------------------------
    if loglevel > 0
        @fprintf("\n")
        @fprintf(" %s\n",repeat("=",80))
        @fprintf(" %s\n","BPdual")
        @fprintf(" %s\n",repeat("=",80))
        @fprintf(" %-20s: %8i %5s"    ,"No. rows"          ,m       ,"")
        @fprintf(" %-20s: %8.2e\n"    ,"λ"                 ,λ          )
        @fprintf(" %-20s: %8i %5s"    ,"No. columns"       ,n       ,"")
        @fprintf(" %-20s: %8.2e\n"    ,"Optimality tol"    ,optTol     )
        @fprintf(" %-20s: %8i %5s"    ,"Maximum iterations",itnMax  ,"")
        @fprintf(" %-20s: %8.2e\n"    ,"Duality tol"       ,gapTol     )
        @fprintf(" %-20s: %8i %5s"    ,"Support trimming"  ,trim,    "")
        @fprintf(" %-20s: %8.2e\n"    ,"Pivot tol"         ,pivTol     )
    end
    
    # ------------------------------------------------------------------
    # Initialize local variables.
    # ------------------------------------------------------------------
    EXIT_STATUS =
        Dict(:exit_optimal=>"optimal solution found -- full Newton step",
             :exit_singular_ls=>"singular least-squares subproblem",
             :exit_iterations=>"too many iterations",
             :exit_infeasible=>"dual infeasible point",
             :exit_requested=>"user requested exit",
             :exit_actmax=>"max no. of active constraints reached",
             :exit_small_gap=>"optimal solution found -- small duality gap",
             :exit_unknown=>"unknown exit")
    eFlag = :exit_unknown
    itn       = 0
    step      = 0.0
    p         = 0                         # index of added constraint
    q         = 0                         # index of deleted constraint
    svar      = ""                        # ...and their string value.
    r         = zeros(m)
    zerovec   = zeros(n)
    nBrks     = 0
    numtrim   = 0
    nprodA    = 0
    nprodAt   = 0
        
    # ------------------------------------------------------------------
    # Cold/warm-start initialization.
    # ------------------------------------------------------------------
    if coldstart
        x = zeros(0)
        z = zeros(n)

    else
        # Trim active-set components that correspond to huge bounds,
        # and then find the min-norm change to y that puts the active
        # constraints exactly on their bounds.
        g = b - λ*y              # Compute steepest-descent dir
        triminf!(active,state,S,R,bl,bu,g)
        nact = length(active)
        x = zeros(nact)
        y = restorefeas(y, active, state, S, R, bl, bu)
        z = A'*y; nprodAt = nprodAt+1

    end

    # Make sure initial point is feasible.
    # bl ≤ z ≤ bu  -->  bl - z ≤ 0
    #              -->  z - bu ≤ 0
    for j in 1:n
        infeasible = (bl[j] - z[j] > feaTol) || (z[j] - bu[j] > feaTol)
        if infeasible
            eFlag = :exit_infeasible
        end
    end
    
    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------
    while true

        g = b - λ*y            # Steepest-descent direction

        condS = condest(R)

        if condS > 1e+10
            eFlag = :exit_singular_ls
            # Pad x with enough zeros to make it compatible with S.
            resize!(x, size(S,2))
            
        else
            x, dy = newtonstep(S, R, g, x, λ)
            
        end
        r = b - S*x

        # ------------------------------------------------------------------
        # Print to log.
        # ------------------------------------------------------------------
        yNorm = norm(y,2)
        rNorm = norm(r,2)
        xNorm = norm(x,1)
        
        pObj, dObj, rGap = objectives(x,y,active,b,bl,bu,λ,rNorm,yNorm)
        nact = length(active)   # size of the working set

        if loglevel > 0
            if q != 0
                svar = @sprintf("%8i%1s", q, "-")
            elseif p != 0
                svar = @sprintf("%8i%1s", p ," ")
            else
                svar = @sprintf("%8s%1s"," "," ")
            end
            if mod(itn,50)==0
                @fprintf("\n %4s  %8s %9s %7s %11s %11s %11s %7s %7s\n",
                            "Itn","step","Add/Drop","Active","rNorm2",
                            "xNorm1","Objective","RelGap","condS")
            end
            @fprintf(
            " %4i  %8.1e %8s %7i %11.4e %11.4e %11.4e %7.1e %7.1e\n",
                    itn, step, svar, nact, rNorm, xNorm,
                    dObj, rGap, condS)
        end

        # ------------------------------------------------------------------
        # Check exit conditions.
        # ------------------------------------------------------------------
        if eFlag == :exit_unknown && rGap < gapTol && nact > 0
            eFlag = :exit_small_gap
        end
        if eFlag == :exit_unknown && itn >= itnMax
            eFlag = :exit_iterations
        end
        if eFlag == :exit_unknown && nact >= actMax
            eFlag = :exit_actmax
        end

        # If this is an optimal solution, trim multipliers before exiting.
        if eFlag == :exit_optimal || eFlag == :exit_small_gap
            if trim == 1
                # Optimal trimming. λ0 may be different from λ.
                # Recompute gradient just in case.
                g = b - λin*y
                trimx!(x,S,R,active,state,g,b,λ,feaTol,optTol,loglevel)
                numtrim = nact - length(active)
                nact = length(active)

            elseif trim == 2
                # Threshold trimming.
                # Not yet implemented.
            end
        end

        # Act on any live exit conditions.
        if eFlag != :exit_unknown
            break
        end

        # --------------------------------------------------------------
        # New iteration starts here.
        # --------------------------------------------------------------
        itn += 1
        p = q = 0
        
        # --------------------------------------------------------------
        # Compute search direction dz.
        # --------------------------------------------------------------
        if norm(dy,Inf) < eps(1.0)       # save a mat-vec prod if dy is 0
            dz = zeros(n)
        else
            dz = A'*dy
            nprodAt += 1
        end

        # ---------------------------------------------
        # Find step to the nearest inactive constraint
        # ---------------------------------------------
        pL, stepL, pU, stepU = step_to_bnd(z, dz, bl, bu, state, tieTol, pivTol)
        
        step = min(1.0, stepL, stepU)
        hitBnd = step < 1.0
        if hitBnd                    # We bumped into a new constraint
            if step == stepL
                p = pL
                state[p] = -1
            else
                p = pU
                state[p] = +1
            end
        end

        y += step*dy                 # Update dual variables.
        if mod(itn,50)==0
            z = A'*y                 # Occasionally set z = A'y directly.
            nprodAt += 1
        else
            z += step*dz
        end


        if hitBnd
            zerovec[p] = 1            # Extract a = A(:,p)
            a = A*zerovec
            nprodA += 1
            zerovec[p] = 0
            R = qraddcol(S,R,a)       # Update R
            S = [S a]                 # Expand S, active
            push!(active, p)
            push!(x, 0)

        else
            #------------------------------------------------------
            # step = 1.  We moved to a minimizer for the current S.
            # See if we can delete an active constraint.
            #------------------------------------------------------
            drop = false
            if length(active) > 0
                dropl = (state[active].==-1) & (x .> +optTol)
                dropu = (state[active].==+1) & (x .< -optTol)
                dropa = dropl | dropu
                drop = any(dropa)
            end
            
            # Delete an active constraint.
            if drop
                nact = length(active)
                qa = indmax(abs(x.*dropa))
                q = active[qa]
                state[q] = 0
                S = S[:, 1:nact .!= qa]
                deleteat!(active, qa)
                deleteat!(x, qa)
                R = qrdelcol(R,qa)

            else
                eFlag = :exit_optimal

            end
            
        end # if step < 1

  end # while true

  tottime = time() - time0

  if loglevel > 0
      @fprintf("\n EXIT BPdual -- %s\n\n",EXIT_STATUS[eFlag])
      @fprintf(" %-20s: %8i %5s","No. significant nnz",sparsity(x),"")
      @fprintf(" %-20s: %8i\n","Products with A",nprodA)
      @fprintf(" %-20s: %8i %5s","No. trimmed nnz",numtrim,"")
      @fprintf(" %-20s: %8i\n","Products with At",nprodAt)
      @fprintf(" %-20s: %8.1e %5s","Solution time (sec)",tottime,"")
      @fprintf("\n")
  end

   # Gather exit data.
   return (x, r, OptimResults(itn,
                              tottime,
                              eFlag,
                              EXIT_STATUS[eFlag],
                              nprodA,
                              nprodAt,
                              numtrim,
                              active, state, y, S, R)
           )
end # function bpdual

# ----------------------------------------------------------------------
# Find step to first bound.
# ----------------------------------------------------------------------
function step_to_bnd(z, dz, bl, bu, state, tieTol, pivTol)

    n = length(dz)
    stepL = stepLtie = stepU = stepUtie = Inf
    pL = pU = 0

    for i in 1:n

        sL = bl[i] -  z[i]
        sU =  z[i] - bu[i]
        
        # skip variables that aren't free.
        state[i] != 0 && continue

        # variable moves to LOWER bound.
        if dz[i] < -pivTol
            tmp = (sL-tieTol)/dz[i]
            if tmp < stepLtie
                stepLtie = tmp
                stepL = max(0, sL/dz[i])
                pL = i
            end
        end

        # variable moves to UPPER bound.
        if dz[i] > +pivTol
            tmp = -(sU-tieTol)/dz[i]
            if tmp < stepUtie
                stepUtie = tmp
                stepU = max(0, -sU/dz[i])
                pU = i
            end
        end
        
    end # if

    return (pL, stepL, pU, stepU)
    
end

# ----------------------------------------------------------------------
# Compute the primal and dual objective values, and the duality gap:
#
#    DP:  minimize_y   - b'y  +  1/2 λ y'y
#         subject to   bl <= A'y <= bu
#
#    PP:  minimize_x   bl'neg(x) + bu'pos(x) + 1/2 λ y'y
#         subject to   Ax + λ y = b.
# ----------------------------------------------------------------------
function objectives(x, y, active,
                    b, bl, bu,
                    λ, rNorm, yNorm)

    bigNum = 1e20
    
    if isempty(x)
        blx = 0.
        bux = 0.
    else
        blx = bl[active]
        blx[blx .< -bigNum] = 0
        blx = dot(blx, min(x,0))
        
        bux = bu[active]
        bux[bux .>  bigNum] = 0
        bux = dot(bux, max(x,0))
    end

    if λ > eps(1.0)
        pObj = blx + bux + rNorm^2/2    # primal objective
    else
        pObj = blx + bux
    end
    dObj  = λ*yNorm^2/2 - dot(b,y)         # dual   objective
    maxpd = max(1, pObj, dObj)
    rGap  = abs(pObj+dObj)/maxpd               # relative duality gap
    
    return (pObj, dObj, rGap)
    
end # function objectives

# ----------------------------------------------------------------------
# Compute a Newton step.  This is a step to a minimizer of the EQP
#
#   min   g'dy + 1/2 λ dy'dy   subj to   S'dy = 0.
#
# The optimality conditions are given by
#
#   [ -λ I   S ] [dy] = [ h ], with  h = b - lam y - S x, 
#   [   S'     ] [dx]   [ 0 ]
#
# where x is an estimate of the Lagrange multiplier. Thus, dx solves
# min ||S dx - h||. On input, g = b - lam y. Alternatively, solve the LS problem
# min ||S  x - g||.
# ----------------------------------------------------------------------
function newtonstep(S, R, g, x, λ)
    m, n = size(S)
    if m==0 || n==0
        dy = g/λ            # Steepest descent
        return (x, dy)
    end
    x, dr = csne(R,S,g)     # LS problem
    if m > n                # Overdetermined system
        dy = dr/λ           # dy is the scaled residual
    else                    # System is square or underdetermined
        dy = zeros(m)       # Anticipate that the residual is 0
    end
    return (x, dy)
end # function newtonstep

# ----------------------------------------------------------------------
# Compute the infeasibility of z relative to bounds bl/bu.
#
#  (bl - z) <= 0  implies  z is   feasible wrt to lower bound
#           >  0  ...           infeasible ...
#  (z - bu) <= 0  implies  z is   feasible wrt to lower bound.
#           >  0  ...           infeasible ...
function infeasibilities(bl::Vector, bu::Vector, z::Vector)
    sL = bl - z
    sU = z - bu
    return (sL, sU)
end

# ----------------------------------------------------------------------
# Trim working constraints with "infinite" bounds.
function triminf!(active::Vector, state::Vector, S::Matrix, R::Matrix,
                  bl::Vector, bu::Vector, g::Vector, b::Vector)
    
    bigbnd = 1e10
    nact = length(active)

    # Generate a list of constraints to trim.
    tlistbl = find( state[active] .== -1 & bl[active] .< -bigbnd )
    tlistbu = find( state[active] .== +1 & bu[active] .> +bigbnd )
    tlist   = [tlistbl; tlistbu]

    if isempty(tlist)
        return
    end

    for q in tlist

        qa = active[q]          # Index of active constraint to delete
        nact = nact - 1 
        S = S[:,1:size(S,2) .!= qa] # Delete column from S
        deleteat!(active, qa)   # Delete index from active set
        R = qrdelcol(R, qa)     # Recompute new QR factorization
        state[q] = 0            # Mark constraint as free
        x = csne(R, S, g)       # Recompute multipliers

        rNorm = norm(b - S*x)
        xNorm = norm(x, 1)
        @fprintf("%4i %4i %4i %10.3e %10.e %10.e %10.e\n",
                k,q,nact,bl[q],bu[q],rNorm,xNorm)
    end
    
end

# ----------------------------------------------------------------------
# n = sparsity(x,threshold) returns the number of elements in x that
# contain most of the weight of the vector.
function sparsity(x::Vector, threshold::Real = 0.9995)
    threshold = clamp(threshold, 0, 1)
    x = sort(abs(x), rev=true)
    tsum = sum(x)
    csum = 0.0
    j = 0
    while j < length(x)
        csum += x[j+1] / tsum
        if csum >= threshold
            j += 1
        end
    end
    return j
end

# ----------------------------------------------------------------------
# Estimate condition number of R'R.
function condest(R::Matrix)
    if isempty(R)
        condS = 1.0
    else
        n = size(R,1)
        rmin = +Inf
        rmax = -Inf
        for i in 1:n
            rii = R[i,i]
            rmin = rii < rmin ? rii : rmin
            rmax = rii > rmax ? rii : rmax
        end
        condS = rmax / rmin
    end
    return condS
end # function condest


# ----------------------------------------------------------------------
# Trim unneeded constraints from the active set.
#
# `trimx` assumes that the current active set is optimal, ie, 1. x has
# the correct sign pattern, and 2. z := A'y is feasible.
#
# Keep trimming the active set until one of these conditions is
# violated. Condition 2 isn't checked directly. Instead, we simply check
# if the resulting change in y, dy, is small.  It would be more
# appropriate to check that dz := A'dy is small, but we want to avoid
# incurring additional products with A. (Implicitly we're assuming that
# A has a small norm.)
function trimx!(x,S,R,active,state,g,b,λ,featol,opttol,loglevel)

    k = 0
    nact = length(active)
    xabs = abs(x)
    xmin, qa = findmin(xabs)
    gNorm = norm(g,Inf)

    while xmin < opttol

        e = sign(x.*(xabs .> opttol)) # Signs of significant multipliers
        q = active[qa] # Index of the corresponding constraint
        a = S[:,qa]    # Save the col from S in case we need to add it back. 
        xsmall = x[qa] # Value of candidate multiplier

        # Trim quantities related to the small multiplier.
        deleteat!(e, qa)
        deleteat!(active, qa)
        S = S[:,1:nact .!= qa]
        R = qrdelcol(R, qa)

        # Recompute the remaining multipliers and their signs.
        (x,dy) = csne(R,S,g)           # min ||g - Sx||_2
        xabs = abs(x)
        et = sign(x.*(xabs .> opttol))
        dyNorm = norm(dy,Inf)/λ        # dy = (g - Sx) / lambda
    
        # Check if the trimmed active set is still optimal
        if any( et .!= e ) || (dyNorm / max(1,gNorm) > featol)
            R = qraddcol(S,R,a)
            S = [S a]
            push!(active, q)
            x = csne(R, S, g)
            break
        end
        
        if loglevel > 0 && mod(k,50)==0
            @fprintf("\n %4s  %8s %8s  %7s  %10s  %10s  %10s\n",
                    "Itn","xSmall","Add/Drop","Active","rNorm2",
                    "xNorm1","dyNorm")
        end

        # The trimmed x is still optimal.
        k += 1
        nact -= 1
        state[q] = 0               # Mark this constraint as free.
        rNorm = norm(b - S*x)
        xNorm = norm(x,1)
        
        # Grab the next canddate multiplier.
        xmin, qa = findmin(xabs)
        
        # Logging.
        if loglevel > 0
            @fprintf(" %4iT %8.1e %8i- %7i  %10.4e  %10.4e  %10.4e\n",
                    k,xsmall,q,nact,rNorm,xNorm,dyNorm)
        end

    end # while

    return (x,S,R,active,state)

end # function trimx

