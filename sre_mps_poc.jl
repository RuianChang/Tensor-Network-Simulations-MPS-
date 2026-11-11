#!/usr/bin/env julia
# OBC Clifford+T SRE proof of concept. Julia 1.12; ITensors + ITensorMPS.
# S2 = -log2[2^-L sum_P (<psi|P|psi>/<psi|psi>)^4].
# Basis: |0>, |1>; site 1 is the MOST significant bit in dense vectors.
# Each layer is ONE alternating brickwork row, followed by T injections.
# cutoff=0, maxdim=typemax(Int): no deliberate Schmidt truncation.
using ITensors, ITensorMPS, LinearAlgebra, Random, Statistics, Printf, Test

const I2 = ComplexF64[1 0; 0 1] 
const X = ComplexF64[0 1; 1 0]
const Y = ComplexF64[0 -im; im 0]
const Z = ComplexF64[1 0; 0 -1]
const T = ComplexF64[1 0; 0 cis(pi/4)]
const H = (X+Z)/sqrt(2)
const PAULIS = [I2,X,Y,Z]
const CNOT = ComplexF64[1 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0]
const SWAP = ComplexF64[1 0 0 0;0 0 1 0;0 1 0 0;0 0 0 1]
const ISWAP = ComplexF64[1 0 0 0;0 0 im 0;0 im 0 0;0 0 0 1]
function cliffords1()
    out = copy(PAULIS)
    for P in (X,Y,Z), sign in (1,-1)
        push!(out,(I2+sign*im*P)/sqrt(2))
    end
    for (P,Q) in ((X,Z),(X,Y),(Y,Z)), sign in (1,-1)
        push!(out,(P+sign*Q)/sqrt(2))
    end
    for signs in ((1,1,1),(1,1,-1),(1,-1,1),(-1,1,1),
                  (1,-1,-1),(-1,1,-1),(-1,-1,1),(-1,-1,-1))
        push!(out,(I2+im*(signs[1]*X+signs[2]*Y+signs[3]*Z))/2)
    end
    out
end
const CLIFFORDS1 = cliffords1()
# Same four local-equivalence classes and weights as inspected history source.
# NOT uniform sampling of the four representatives themselves.
function random_clifford(rng)
    left = kron(rand(rng,CLIFFORDS1),rand(rng,CLIFFORDS1))
    right = kron(rand(rng,CLIFFORDS1),rand(rng,CLIFFORDS1))
    u = rand(rng)
    rep = u < .05 ? kron(I2,I2) : u < .10 ? SWAP : u < .55 ? ISWAP : CNOT
    left*rep*right
end
struct Layer
    gates::Vector{Tuple{Int,Matrix{ComplexF64}}}
    t_sites::Vector{Int}
end
function sample_plan(L,D,pT,seed; boundary_t=false)
    rng=MersenneTwister(seed)
    [Layer([(j,random_clifford(rng)) for j in (isodd(d) ? 1 : 2):2:(L-1)],
           [j for j in 1:L if rand(rng)<pT &&
            (boundary_t || isodd(d) || (j!=1 && j!=L))]) for d in 1:D]
end
# Explicit loops prevent accidental reversal of the two-qubit basis by reshape.
function gate_tensor(U,s1,s2)
    g=ITensor(ComplexF64,prime(s1),prime(s2),s1,s2)
    for a in 0:1,b in 0:1,c in 0:1,e in 0:1
        g[prime(s1)=>a+1,prime(s2)=>b+1,s1=>c+1,s2=>e+1]=U[2a+b+1,2c+e+1]
    end
    g
end
function evolve(psi,layer)
    s=siteinds(psi)
    for (j,U) in layer.gates
        psi=apply(gate_tensor(U,s[j],s[j+1]),psi;cutoff=0.0,maxdim=typemax(Int))
    end
    for j in layer.t_sites
        psi=apply(ITensor(T,prime(s[j]),s[j]),psi;cutoff=0.0,maxdim=typemax(Int))
    end
    psi
end
# Return A[left,physical,right], with unit OBC legs at the ends.
function local_array(psi,j)
    s=siteind(psi,j); L=length(psi)
    idx=Index[]
    j>1 && push!(idx,linkind(psi,j-1))
    push!(idx,s)
    j<L && push!(idx,linkind(psi,j))
    dl=j==1 ? 1 : dim(linkind(psi,j-1))
    dr=j==L ? 1 : dim(linkind(psi,j))
    reshape(ComplexF64.(Array(psi[j],idx...)),dl,2,dr)
end
# A stable OBC norm contraction; never first form the exponentially small norm.
function log_norm2(psi)
    E=ones(ComplexF64,1,1); logscale=0.0
    for j in 1:length(psi)
        A=local_array(psi,j); dl,_,dr=size(A)
        F=zeros(ComplexF64,dr,dr)
        for s in 1:2
            M=A[:,s,:]; F .+= M'*E*M
        end
        r=norm(F); isfinite(r)&&r>0 || error("Invalid norm environment at site $j")
        E=F/r; logscale+=log(r)
    end
    v=only(E); abs(imag(v))<1e-9*abs(v) || error("Complex norm")
    real(v)>0 || error("Nonpositive norm")
    logscale+log(real(v))
end
kron4(mats)=kron(kron(kron(mats[1],mats[2]),mats[3]),mats[4])
# Lambda = (I^⊗4+X^⊗4+Y^⊗4+Z^⊗4)/2 = 2*projector.
# Its rank-four square root is represented by even-parity complement pairs.
# b_s = tensor_product_r A[s_r] + tensor_product_r A[1-s_r],
# s=(0000,0011,0101,0110). Thus sum b_s† b_s inserts Lambda,
# and contracting the replica-MPS norm gives Z2 with NO extra 2^-L factor.
const EVEN_REPS=((0,0,0,0),(0,0,1,1),(0,1,0,1),(0,1,1,0))
function sre_replica(psi;max_work_gib=2.0)
    E=ones(ComplexF64,1,1); logs=0.0; peak_est=0.0
    for j in 1:length(psi)
        A=local_array(psi,j); dl,_,dr=size(A)
        nl=big(dl)^4; nr=big(dr)^4
        # Conservative live-array estimate, excluding BLAS/runtime overhead.
        bytes=16*(nl^2+2nr^2+4nl*nr)
        peak_est=max(peak_est,Float64(bytes))
        bytes <= max_work_gib*2.0^30 || error("Replica workspace estimate $(Float64(bytes)/2.0^30) GiB exceeds limit at site $j (chi=$dl,$dr). Reduce depth or raise --max-work-gib.")
        M=(A[:,1,:],A[:,2,:])
        # Scale A before forming its fourth power to control intermediate ranges.
        a=norm(A); isfinite(a)&&a>0 || error("Invalid MPS tensor")
        M=(M[1]/a,M[2]/a); logs+=8log(a)
        F=zeros(ComplexF64,Int(nr),Int(nr))
        for bits in EVEN_REPS
            B=kron4([M[b+1] for b in bits])+kron4([M[2-b] for b in bits])
            F .+= B'*E*B
        end
        r=norm(F); isfinite(r)&&r>0 || error("Invalid replica environment at site $j")
        E=F/r; logs+=log(r)
    end
    v=only(E)
    abs(imag(v))<=1e-8*abs(v) || error("Replica scalar has imaginary residue")
    real(v)>0 || error("Replica scalar nonpositive")
    lnN=log_norm2(psi)
    lnZ=logs+log(real(v))-4lnN
    (sre=-lnZ/log(2),logZ=lnZ,lognorm2=lnN,workspace_est_bytes=peak_est)
end
function dense_evolve(v,layer,L)
    w=copy(v)
    for (j,U) in layer.gates
        m1=1<<(L-j); m2=1<<(L-j-1)
        for b in 0:(length(w)-1)
            (b & (m1|m2))==0 || continue
            idx=[b,b|m2,b|m1,b|m1|m2].+1
            w[idx]=U*w[idx]
        end
    end
    for j in layer.t_sites,b in 0:(length(w)-1)
        (b & (1<<(L-j)))!=0 && (w[b+1]*=cis(pi/4))
    end
    w
end
# Independent exhaustive Pauli oracle (small L only).
function dense_sre(v,L)
    L<=6 || error("Dense oracle is limited to L<=6")
    v=v/norm(v); total=0.0
    for labels in Iterators.product(ntuple(_->0:3,L)...)
        P=reduce(kron,[PAULIS[a+1] for a in labels])
        ev=dot(v,P*v)
        abs(imag(ev))<1e-9 || error("Nonreal Pauli expectation")
        total+=real(ev)^4
    end
    -log2(total/2.0^L)
end
# Replay file has comments '# L=... D=...', G<TAB>layer<TAB>site<TAB>
# row-major interleaved real/imag matrix entries (32 floats), and T<TAB>layer<TAB>site.
function write_plan(path,plan,L)
    open(path,"w") do io
        println(io,"# L=$L D=$(length(plan)); sites and layers are 1-based; basis 00,01,10,11")
        for (d,layer) in enumerate(plan)
            for (j,U) in layer.gates
                fields=String["G",string(d),string(j)]
                for a in 1:4,b in 1:4
                    append!(fields,[repr(real(U[a,b])),repr(imag(U[a,b]))])
                end
                println(io,join(fields,'\t'))
            end
            for j in layer.t_sites; println(io,"T\t$d\t$j"); end
        end
    end
end
function read_plan(path,L,D)
    plan=[Layer(Tuple{Int,Matrix{ComplexF64}}[],Int[]) for _ in 1:D]
    for line in eachline(path)
        if startswith(line,"#")
            header=match(r"L=(\d+) D=(\d+)",line)
            if header !== nothing
                parse(Int,header[1])==L && parse(Int,header[2])==D || error("Replay L/depth does not match options")
            end
        end
        (isempty(strip(line))||startswith(line,"#")) && continue
        f=split(line,'\t'); kind=f[1]; d=parse(Int,f[2]); j=parse(Int,f[3])
        1<=d<=D || error("Invalid layer $d")
        if kind=="G"
            1<=j<L || error("Invalid pair $j"); length(f)==35 || error("Bad gate row")
            x=parse.(Float64,f[4:end]); U=zeros(ComplexF64,4,4)
            for a in 1:4,b in 1:4; k=2*(4*(a-1)+b)-1; U[a,b]=complex(x[k],x[k+1]); end
            norm(U'*U-I)<1e-10 || error("Replay gate is not unitary")
            push!(plan[d].gates,(j,U))
        elseif kind=="T"
            1<=j<=L || error("Invalid site"); push!(plan[d].t_sites,j)
        else
            error("Unknown record $kind")
        end
    end
    plan
end
function selftest()
    @testset "Replica SRE and exact evolution" begin
        for L in (1,2,4,6)
            s=siteinds("Qubit",L); psi=MPS(s,fill("0",L))
            @test abs(sre_replica(psi).sre)<1e-10
        end
        s=siteinds("Qubit",4); psi=MPS(s,fill("0",4))
        for j in 1:4
            psi=apply(ITensor(H,prime(s[j]),s[j]),psi;cutoff=0.0)
            psi=apply(ITensor(T,prime(s[j]),s[j]),psi;cutoff=0.0)
        end
        @test sre_replica(psi).sre ≈ 4log2(4/3) atol=1e-10
        psi[1]*=0.13+0.2im
        @test sre_replica(psi).sre ≈ 4log2(4/3) atol=1e-10
        for L in (2,4,6), p in (0.0,0.5,1.0), seed in (13,72)
            psi=MPS(siteinds("Qubit",L),fill("0",L)); v=zeros(ComplexF64,2^L); v[1]=1
            for layer in sample_plan(L,3,p,seed)
                psi=evolve(psi,layer); v=dense_evolve(v,layer,L)
                @test sre_replica(psi).sre ≈ dense_sre(v,L) atol=2e-9
                # Test actual wavefunction, not just its Clifford-invariant SRE.
                tensor=reduce(*,psi); sites=siteinds(psi)
                vmps=vec(Array(tensor,reverse(sites)...))
                @test vmps ≈ v atol=2e-10
            end
        end
        # A non-circuit random entangled MPS probes complex conjugation / Y signs.
        rng=MersenneTwister(9); s=siteinds("Qubit",4)
        v=randn(rng,ComplexF64,16); v/=norm(v)
        psi=MPS(ITensor(reshape(v,2,2,2,2),reverse(s)...),s;cutoff=0.0)
        @test sre_replica(psi).sre ≈ dense_sre(v,4) atol=2e-9
        s=siteinds("Qubit",4000); psi=MPS(s,fill("0",4000))
        for j in 1:4000
            psi[j]=noprime(ITensor(T*H,prime(s[j]),s[j])*psi[j])
        end
        @test (3/4)^4000 == 0.0
        @test sre_replica(psi).sre ≈ 4000log2(4/3) atol=1e-7
        plan=sample_plan(4,3,0.5,7)
        mktemp() do path,io
            close(io); write_plan(path,plan,4); loaded=read_plan(path,4,3)
            @test all(loaded[d].t_sites==plan[d].t_sites && loaded[d].gates==plan[d].gates for d in 1:3)
        end
    end
end
function main(args=ARGS)
    if "--self-test" in args; selftest(); return; end
    cfg=Dict("L"=>"4","depth"=>"4","pT"=>"0.5","samples"=>"2",
             "seed"=>"20261003","out"=>"benchmark","boundary-t"=>"false",
             "dense-check"=>"true","max-work-gib"=>"2","plan"=>"","save-plans"=>"false")
    if "--help" in args
        println("julia --project=. sre_mps_poc.jl [--L 4 --depth 4 --pT 0.5 --samples 2 --seed 20261003 --out benchmark --boundary-t false --dense-check true --max-work-gib 2 --plan replay.tsv --save-plans false]\n--self-test runs independent validation. No truncation; OBC only. Dense check requires L<=6.\n--save-plans true writes a replay TSV for each realization (default: false).")
        return
    end
    iseven(length(args)) || error("Options are --key value pairs")
    for k in 1:2:length(args)
        key=replace(args[k],r"^--"=>""); haskey(cfg,key)||error("Unknown option"); cfg[key]=args[k+1]
    end
    L=parse(Int,cfg["L"]); D=parse(Int,cfg["depth"]); ns=parse(Int,cfg["samples"])
    p=parse(Float64,cfg["pT"]); seed=parse(Int,cfg["seed"]); prefix=cfg["out"]
    L>=1&&D>=0&&ns>=1&&0<=p<=1 || error("Invalid sizes / probability")
    dense=parse(Bool,cfg["dense-check"]); boundary=parse(Bool,cfg["boundary-t"])
    save_plans=parse(Bool,cfg["save-plans"])
    dense&&L>6 && error("Use --dense-check false for L>6")
    !isempty(cfg["plan"])&&ns!=1 && error("Replay requires --samples 1")
    mkpath(dirname(abspath(prefix)))
    BLAS.set_num_threads(1)
    # Warmup avoids charging first compilation to the layer benchmark.
    tmp=MPS(siteinds("Qubit",2),fill("0",2))
    tmp=evolve(tmp,sample_plan(2,1,1.0,1)[1]); sre_replica(tmp)
    open(prefix*"_metadata.txt","w") do io
        println(io,"Julia=$(VERSION)\nITensors=$(pkgversion(ITensors))\nITensorMPS=$(pkgversion(ITensorMPS))\nL=$L\nD=$D\npT=$p\nsamples=$ns\nseed=$seed\nOBC=true\nboundary_t=$boundary\nsave_plans=$save_plans\ncutoff=0\nmaxdim=typemax(Int)\nBLAS_threads=1\nRNG=MersenneTwister; not NumPy seed-equivalent\nreplica_bond=chi^4; no Klein-group virtual compression\nworkspace estimate excludes runtime/BLAS overhead\nreplay_source=$(cfg["plan"])\n")
    end
    open(prefix*"_layers.csv","w") do io
        println(io,"realization,seed,layer,L,pT,sre_bits,sre_density,accumulated_sre,chi_max,chi_median,replica_chi_max,norm2_before_normalize,evolution_seconds,sre_seconds,sre_alloc_bytes,workspace_est_bytes,dense_sre,abs_error")
        for n in 1:ns
            plan=isempty(cfg["plan"]) ? sample_plan(L,D,p,seed+n-1;boundary_t=boundary) : read_plan(cfg["plan"],L,D)
            save_plans && write_plan(prefix*"_plan_$(n).tsv",plan,L)
            psi=MPS(siteinds("Qubit",L),fill("0",L)); v=zeros(ComplexF64,dense ? 2^L : 0)
            dense && (v[1]=1); accumulated=0.0
            for d in 0:D
                elapsed=0.0; before=1.0
                if d>0
                    evo=@timed evolve(psi,plan[d]); psi=evo.value; elapsed=evo.time
                    before=exp(log_norm2(psi)); normalize!(psi)
                    dense && (v=dense_evolve(v,plan[d],L))
                end
                measure=@timed sre_replica(psi;max_work_gib=parse(Float64,cfg["max-work-gib"]))
                m=measure.value; chi=L==1 ? [1] : [dim(linkind(psi,j)) for j in 1:L-1]
                d>0 && (accumulated+=m.sre)
                exact=dense ? dense_sre(v,L) : NaN; err=abs(m.sre-exact)
                dense && err>2e-9 && error("Dense/replica mismatch $err")
                fields=(n,isempty(cfg["plan"]) ? string(seed+n-1) : "replay",d,L,p,m.sre,m.sre/L,accumulated,maximum(chi),median(chi),big(maximum(chi))^4,before,elapsed,measure.time,measure.bytes,m.workspace_est_bytes,exact,err)
                println(io,join(fields,',')); flush(io)
                @printf("sample=%d layer=%d SRE=%.12f chi=%d replica_time=%.4fs\n",n,d,m.sre,maximum(chi),measure.time)
            end
        end
    end
    println("Saved $(prefix)_layers.csv" * (save_plans ? " and explicit replay plans." : "."))
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end