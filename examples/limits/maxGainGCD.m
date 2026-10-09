%% Maximum broadside gain of a copper 2:1 plate fed by a 50 Ohm port, dual bound with GCD
% Maximizes the gain G = 4*pi*U/Pin in the broadside direction (theta = 0,
% both polarizations) of a copper structure inside a 2:1 rectangle, ka = 1,
% f = 2.45 GHz, fed by a delta gap at the basis function nearest the
% midpoint of the longer edge. The input voltage 1 V and the input
% impedance 50 Ohm fix the input current I0 = 20 mA and the input power
% Pin = 10 mW, so the bound is on the realized gain of a matched antenna.
%
% The dual bound with GCD refinement is evaluated by maxGainGCD.py in the
% current variable x = I/I0 in the RWG basis with the non-diagonal material
% operator Z = Z0 + Zs*L (main.tex, Sec. "Inference with Non-Diagonal
% Material Operator"). The density d of each basis function is inferred
%   pointwise:   d_i = clip(Re[conj(b_i) x_i]/|b_i|^2), b_i = u_i/(Zs L_ii),
%                u = v - (Z - Zs diag(L_ii)) x
%   Lagrangian:  min_d (x(d) - xopt)' Lstar (x(d) - xopt),
%                x(d) = (Lambda + D Z')^{-1} D v (bounded trust region,
%                started from the pointwise fit)
% and binarized by a threshold sweep (port always kept). The design is the
% threshold of the Lagrangian-weighted solution fit with the largest
% realized gain (1 - |Gamma|^2) G into Zin0, the quantity bounded by the dual.
%
% stage = 'assemble' : mesh, normalized operators, port -> *_in.mat
% stage = 'dual'     : run maxGainGCD.py -> *_out.mat
% stage = 'post'     : *_out.mat -> inferred design, gain, figures
% stage = 'greedy'   : binary local search from the inferred design
%                      (single basis function flips, steepest ascent in the
%                      realized gain, port kept) -> *_greedy.mat, figures
% stage = 'all'      : all of the above (default)
%
% Requires AToM (with the TopologySensitivity package) on the path and a
% Python environment with dolphindes (pythonExe).

if ~exist('stage', 'var')
    stage = 'all';
end
close all;

%% Settings
if ~exist('m', 'var')
    m = 25;                 % EuCAP27 ortho mesh, 25 -> N = 285, 50 -> N = 744
end
if ~exist('Ngcd', 'var')
    Ngcd = 20;              % number of GCD iterations
end
tag = sprintf('maxGain_patch21ka1_gcd%d', Ngcd);
if m ~= 25
    tag = sprintf('%s_m%d', tag, m);
end
ka          = 1;
fixMeshToKa = 2;
f0          = 2.45e9;       % frequency (Hz)
sigma       = 5.96e7;       % copper conductivity (S/m)
Vp          = 1;            % input voltage (V)
Zin0        = 50;           % required input impedance (Ohm)
I0          = Vp/Zin0;      % input current (A)
Pin         = Vp*I0/2;      % input power (W)
thetaDir    = 0;            % gain direction (broadside)
phiDir      = 0;
quadOrder   = 7;

thisDir   = fileparts(mfilename('fullpath'));
pythonExe = fullfile(thisDir, '..', '..', '.venv', 'bin', 'python');
pyScript  = fullfile(thisDir, 'maxGainGCD.py');
inFile    = fullfile(thisDir, [tag '_in.mat']);
outFile   = fullfile(thisDir, [tag '_out.mat']);
opFile    = fullfile(thisDir, [tag '_operators.mat']);

%% Physical constants
c0    = models.utilities.constants.c0;
Z0vac = models.utilities.constants.Z0;
mu0   = models.utilities.constants.mu0;
k0    = 2*pi*f0/c0;
lam   = c0/f0;
delta = sqrt(2/(2*pi*f0*mu0*sigma));  % skin depth
Zs    = (1 + 1i)/(sigma*delta);       % surface impedance

%% Mesh (EuCAP27/code/getMatricesFixedMesh.m)
a    = ka/k0;
len  = 4/sqrt(5)*a;
wid  = len/2;
mEff = m*fixMeshToKa/ka;
Ny   = ceil((wid/lam*mEff + 1)/2);
[~, ~, meshPx] = models.utilities.meshPublic.pixelGridToOrthoMesh(2*ones(Ny, 2*Ny), 1);
nodes = meshPx.nodes./meshPx.normDistanceA*sqrt((wid/2)^2 + (len/2)^2);
mesh  = models.utilities.meshPublic.getMeshData2D(nodes, meshPx.connectivityList);
h     = len/(2*Ny);         % pixel size

if any(strcmp(stage, {'assemble', 'all'}))
    %% Operators (normalized: Ohm, port voltage in V, terminal current in A)
    BF   = models.solvers.MoM2D.basisFcns.getBasisFcns(mesh);
    N    = BF.nUnknowns;
    lmax = models.utilities.matrixOperators.MoM2D.SMatrix.lmax(k0, mesh.nodes);
    [OP, ~] = models.utilities.matrixOperators.MoM2D.batch.evaluate(mesh, f0, 0, ...
        'normalize', false, 'symmetrize', true, 'quadOrder', quadOrder, ...
        'lmax', lmax, 'requests', {'R0', 'X0'});
    Lmat = full(models.utilities.matrixOperators.MoM2D.ohmicLosses.computeL(mesh, BF));
    [~, Fph, Fth] = models.utilities.matrixOperators.MoM2D.farfield.computeU( ...
        mesh, BF, f0, thetaDir, phiDir, 'total', quadOrder);
    dn   = 1./mesh.triangleEdgeLengths(BF.data(:, 3));   % normalization 1/l_n
    R0   = dn.*real(OP.R0).*dn.';   R0   = (R0 + R0.')/2;
    X0   = dn.*real(OP.X0).*dn.';   X0   = (X0 + X0.')/2;
    Lmat = dn.*Lmat.*dn.';          Lmat = (Lmat + Lmat.')/2;
    Fth  = Fth(:).'.*dn.';
    Fph  = Fph(:).'.*dn.';

    %% Port: BF nearest the midpoint of the longer (bottom, y = -wid/2) edge
    feederPosition = [0, min(mesh.nodes(:, 2)) + h/2, 0];
    edgeCenters = mesh.triangleEdgeCenters(BF.data(:, 3), :);
    [dist, port] = min(models.utilities.vectorOperations.rowNorm(edgeCenters - feederPosition));
    fprintf('N = %d, size %.4f x %.4f m, Zs = %.4e%+.4ej Ohm, port BF %d at [%.4f %.4f] (%.1e m)\n', ...
        N, len, wid, real(Zs), imag(Zs), port, edgeCenters(port, 1:2), dist);

    % full plate fed at the port (reference)
    Z = R0 + 1i*X0 + Zs*Lmat;
    V = zeros(N, 1); V(port) = Vp;
    Ifull = Z\V;
    [Gfull, ZinFull] = portGain(Ifull, port, Vp, Fth, Fph, Z0vac);
    fprintf('full plate: Zin = %.3f %+.3fj Ohm, G = %.4g, realized G = %.4g\n', ...
        real(ZinFull), imag(ZinFull), Gfull, Gfull*(1 - abs((ZinFull - Zin0)/(ZinFull + Zin0))^2));

    port = port - 1; %#ok<NASGU> zero based in Python
    save(inFile, 'R0', 'X0', 'Lmat', 'Zs', 'Fth', 'Fph', 'port', 'I0', 'Vp', 'Ngcd');
    port = port + 1;
    save(opFile, 'mesh', 'BF', 'R0', 'X0', 'Lmat', 'Zs', 'Fth', 'Fph', 'port', 'Gfull', 'ZinFull');
    fprintf('saved %s\n', inFile);
end

if any(strcmp(stage, {'dual', 'all'}))
    %% Dual bound with GCD (Python)
    cmd = sprintf('"%s" -u "%s" "%s" "%s"', pythonExe, pyScript, inFile, outFile);
    fprintf('Running %s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('maxGainGCD.py failed (status %d).', status);
    end
end

if any(strcmp(stage, {'post', 'all'}))
    %% Dual optimum
    ops = load(opFile);
    out = load(outFile);
    BF = ops.BF; R0 = ops.R0; X0 = ops.X0; Lmat = ops.Lmat; port = ops.port;
    Fth = ops.Fth; Fph = ops.Fph;
    N  = BF.nUnknowns;
    Z  = R0 + 1i*X0 + Zs*Lmat;
    V  = zeros(N, 1); V(port) = Vp;
    v  = V/I0;
    xopt  = out.xopt(:);
    Lstar = (out.Lstar + out.Lstar')/2;
    Glb   = out.GlbGCD;
    fprintf('bounds: global %.4g, GCD %.4g, local %.4g, arbitrary current %.4g, full plate %.4g\n', ...
        out.GlbGlobal, out.GlbGCD, out.GlbLocal, out.gMax, ops.Gfull);

    %% Inference: pointwise current fit
    Lam  = Zs*diag(Lmat);                % self material term Zs L_ii
    Zp   = Z - diag(Lam);                % Z' = Z - Lambda
    free = setdiff((1:N).', port);
    u    = v - Zp*xopt;
    b    = u./Lam;
    dPw  = min(max(real(conj(b).*xopt)./abs(b).^2, 0), 1);
    dPw(port) = 1;

    %% Inference: Lagrangian-weighted solution fit
    [W, E] = eig(Lstar, 'vector');
    C = sqrt(max(E, 0)).*W';
    fun  = @(df) lagrangianResidual(df, free, port, Lam, Zp, v, xopt, C);
    opts = optimoptions('lsqnonlin', 'Algorithm', 'trust-region-reflective', ...
        'SpecifyObjectiveGradient', true, 'MaxIterations', 2000, ...
        'FunctionTolerance', 1e-12, 'StepTolerance', 1e-10, 'Display', 'final');
    t1 = tic;
    dfree = lsqnonlin(fun, dPw(free), zeros(N - 1, 1), ones(N - 1, 1), opts);
    dLag = ones(N, 1); dLag(free) = dfree;
    Jpw  = sum(fun(dPw(free)).^2);
    Jlag = sum(fun(dfree).^2);
    fprintf('J = (x(d) - xopt)''L*(x(d) - xopt): pointwise %.4g, Lagrangian fit %.4g (%.1f s)\n', ...
        Jpw, Jlag, toc(t1));
    [Gpw, ZinPw]   = portGain(I0*realizedCurrent(dPw, Lam, Zp, v), port, Vp, Fth, Fph, Z0vac);
    [Glag, ZinLag] = portGain(I0*realizedCurrent(dLag, Lam, Zp, v), port, Vp, Fth, Fph, Z0vac);
    fprintf('relaxed density: pointwise G = %.4g (Zin %.1f%+.1fj), Lagrangian G = %.4g (Zin %.1f%+.1fj)\n', ...
        Gpw, real(ZinPw), imag(ZinPw), Glag, real(ZinLag), imag(ZinLag));

    %% Binarization: threshold sweep of both densities
    sweep = struct('name', {'pointwise', 'Lagrangian'}, 'd', {dPw, dLag});
    for iS = 1:numel(sweep)
        d = sweep(iS).d;
        [~, order] = sort(d(free), 'descend');
        order = free(order);
        nList = 1:N;
        [G, Greal, Jb] = deal(nan(1, N));
        Zin = nan(1, N);
        for iN = nList
            S = sort([port; order(1:iN - 1)]);
            I = zeros(N, 1);
            I(S) = Z(S, S)\V(S);
            [G(iN), Zin(iN)] = portGain(I, port, Vp, Fth, Fph, Z0vac);
            Greal(iN) = G(iN)*(1 - abs((Zin(iN) - Zin0)/(Zin(iN) + Zin0))^2);
            Jb(iN) = real((I/I0 - xopt)'*Lstar*(I/I0 - xopt));
        end
        [~, iBest] = max(Greal);
        sweep(iS).order = order; sweep(iS).nList = nList; sweep(iS).G = G;
        sweep(iS).Greal = Greal; sweep(iS).Zin = Zin; sweep(iS).J = Jb; sweep(iS).iBest = iBest;
        [GmaxS, iG] = max(G);
        fprintf(['%-10s sweep: best realized %d/%d BFs, realized G = %.4g, G = %.4g, Zin = %.2f %+.2fj, ' ...
            'J = %.4g; max G = %.4g (%d BFs)\n'], sweep(iS).name, nList(iBest), N, Greal(iBest), ...
            G(iBest), real(Zin(iBest)), imag(Zin(iBest)), Jb(iBest), GmaxS, nList(iG));
    end
    best  = sweep(2);                    % Lagrangian-weighted solution fit
    S     = sort([port; best.order(1:best.iBest - 1)]);
    Ibest = zeros(N, 1); Ibest(S) = Z(S, S)\V(S);
    Gbest = best.G(best.iBest); ZinBest = best.Zin(best.iBest); GrBest = best.Greal(best.iBest);
    fprintf(['inferred design (Lagrangian fit): %d/%d BFs, realized G = %.4g (%.2f dBi), G = %.4g, ' ...
        'Zin = %.2f %+.2fj, bound %.4g (%.2f dBi), full plate %.4g (realized %.4g)\n'], ...
        numel(S), N, GrBest, 10*log10(GrBest), Gbest, real(ZinBest), imag(ZinBest), Glb, ...
        10*log10(Glb), ops.Gfull, ops.Gfull*(1 - abs((ops.ZinFull - Zin0)/(ops.ZinFull + Zin0))^2));

    save(fullfile(thisDir, [tag '_post.mat']), 'dPw', 'dLag', 'Jpw', 'Jlag', 'sweep', 'S', ...
        'Ibest', 'Gbest', 'GrBest', 'ZinBest', 'Gpw', 'Glag');

    %% Figures
    % density per triangle (mean over its basis functions)
    nT = size(mesh.connectivityList, 1);
    triDens = @(d) accumarray([BF.data(:, 2); BF.data(:, 4)], [d; d], [nT 1]) ./ ...
        max(accumarray([BF.data(:, 2); BF.data(:, 4)], 1, [nT 1]), 1);
    pc = mesh.triangleEdgeCenters(BF.data(port, 3), :);
    figure('Position', [100 100 900 600]);
    dens = {dPw, dLag};
    ttl  = {sprintf('pointwise fit, G(d) = %.3g', Gpw), ...
        sprintf('Lagrangian fit, J = %.3g, G(d) = %.3g', Jlag, Glag)};
    for iD = 1:2
        subplot(2, 1, iD);
        patch('Faces', mesh.connectivityList, 'Vertices', mesh.nodes, ...
            'FaceVertexCData', triDens(dens{iD}), 'FaceColor', 'flat', 'EdgeColor', 'none');
        hold on; plot(pc(1), pc(2), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
        axis equal off; colormap(flipud(gray)); clim([0 1]); colorbar;
        title(['density d, ' ttl{iD}]);
    end
    exportgraphics(gcf, fullfile(thisDir, [tag '_inference.png']));

    figure;
    hS = gobjects(1, 2*numel(sweep));
    col = lines(numel(sweep));
    for iS = 1:numel(sweep)
        hS(2*iS - 1) = plot(sweep(iS).nList, sweep(iS).G, '-', 'Color', col(iS, :)); hold on;
        hS(2*iS) = plot(sweep(iS).nList, sweep(iS).Greal, '.-', 'Color', col(iS, :));
    end
    plot(best.nList(best.iBest), GrBest, 'ko', 'MarkerSize', 10, 'LineWidth', 1.5);
    yline(out.GlbGCD, 'r--', sprintf('GCD bound %.3g', out.GlbGCD));
    yline(out.GlbGlobal, 'k:', sprintf('global bound %.3g', out.GlbGlobal));
    if isfinite(out.GlbLocal), yline(out.GlbLocal, 'b-.', sprintf('local bound %.3g', out.GlbLocal)); end
    yline(ops.Gfull, 'g-', sprintf('full plate %.3g', ops.Gfull));
    xlabel('number of metal BFs (sorted by density)'); ylabel('broadside gain');
    legend(hS, {'pointwise G', 'pointwise (1-|\Gamma|^2)G', 'Lagrangian G', ...
        'Lagrangian (1-|\Gamma|^2)G'}, 'Location', 'northwest'); grid on;
    exportgraphics(gcf, fullfile(thisDir, [tag '_sweep.png']));

    TSaux.plotStructure(mesh, BF, port, S, false);
    title(sprintf('inferred: %d/%d BFs, G_r = %.3g (%.2f dBi), bound %.3g, Z_{in} = %.1f %+.1fj \\Omega', ...
        numel(S), N, GrBest, 10*log10(GrBest), Glb, real(ZinBest), imag(ZinBest)));
    exportgraphics(gcf, fullfile(thisDir, [tag '_structure.png']));

    results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', Ibest, ...
        'part', 'abs', 'arrowScale', 'proportional');
    title(sprintf('|J| of the inferred structure, G = %.3g', Gbest));
    exportgraphics(gcf, fullfile(thisDir, [tag '_structureCurrent.png']));

    results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', out.Iopt(:), ...
        'part', 'abs', 'arrowScale', 'proportional');
    title(sprintf('|J_{opt}| (GCD), G = %.3g, bound %.3g', out.GIopt, Glb));
    exportgraphics(gcf, fullfile(thisDir, [tag '_dualCurrent.png']));

    % gain pattern in the xz (phi = 0) and yz (phi = 90 deg) planes
    th = linspace(-pi/2, pi/2, 61);
    figure;
    planes = [0 pi/2];
    for iP = 1:2
        Gcut = zeros(size(th));
        for it = 1:numel(th)
            [~, fph, fth] = models.utilities.matrixOperators.MoM2D.farfield.computeU( ...
                mesh, BF, f0, abs(th(it)), planes(iP) + pi*(th(it) < 0), 'total', quadOrder);
            dn = 1./mesh.triangleEdgeLengths(BF.data(:, 3));
            Gcut(it) = portGain(Ibest, port, Vp, fth(:).'.*dn.', fph(:).'.*dn.', Z0vac);
        end
        polarplot(th, Gcut, 'LineWidth', 1.5); hold on;
    end
    polarplot(0, Glb, 'r*', 'MarkerSize', 10);
    pax = gca; pax.ThetaZeroLocation = 'top'; pax.ThetaDir = 'clockwise';
    pax.ThetaLim = [-90 90];
    legend({'xz plane', 'yz plane', 'GCD bound'}, 'Location', 'southoutside');
    title('gain of the inferred design');
    exportgraphics(gcf, fullfile(thisDir, [tag '_pattern.png']));
end

if any(strcmp(stage, {'greedy', 'all'}))
    %% Greedy binary local search from the inferred design
    ops  = load(opFile);
    out  = load(outFile);
    post = load(fullfile(thisDir, [tag '_post.mat']));
    BF = ops.BF; port = ops.port; Fth = ops.Fth; Fph = ops.Fph;
    N  = BF.nUnknowns;
    Z  = ops.R0 + 1i*ops.X0 + Zs*ops.Lmat;
    V  = zeros(N, 1); V(port) = Vp;
    Glb = out.GlbGCD;
    keep = false(N, 1); keep(post.S) = true;
    free = setdiff((1:N).', port);
    evalDesign = @(keep) designGain(keep, Z, V, port, Vp, Fth, Fph, Z0vac, Zin0);
    [Gr, G, Zin] = evalDesign(keep);
    fprintf('greedy start: %d BFs, realized G = %.4g, G = %.4g, Zin = %.2f %+.2fj\n', ...
        nnz(keep), Gr, G, real(Zin), imag(Zin));
    hist = [0 Gr G];
    maxPass = 500;
    t1 = tic;
    for iPass = 1:maxPass
        GrFlip = -inf(N, 1);
        for j = free.'
            k = keep; k(j) = ~k(j);
            GrFlip(j) = evalDesign(k);
        end
        [GrNew, jBest] = max(GrFlip);
        if GrNew <= Gr*(1 + 1e-9)
            break;
        end
        keep(jBest) = ~keep(jBest);
        [Gr, G, Zin] = evalDesign(keep);
        hist(end + 1, :) = [iPass Gr G]; %#ok<AGROW>
        fprintf('pass %3d: flip BF %3d (%s), %d BFs, realized G = %.4g, G = %.4g, Zin = %.2f %+.2fj\n', ...
            iPass, jBest, string(ifelse(keep(jBest), 'add', 'remove')), nnz(keep), Gr, G, real(Zin), imag(Zin));
    end
    fprintf('greedy done (%d flips, %.1f s): %d/%d BFs, realized G = %.4g (%.2f dBi), G = %.4g, bound %.4g\n', ...
        size(hist, 1) - 1, toc(t1), nnz(keep), N, Gr, 10*log10(Gr), G, Glb);
    Sg = find(keep);
    Ig = zeros(N, 1); Ig(Sg) = Z(Sg, Sg)\V(Sg);
    save(fullfile(thisDir, [tag '_greedy.mat']), 'Sg', 'Ig', 'Gr', 'G', 'Zin', 'hist');

    TSaux.plotStructure(mesh, BF, port, Sg, false);
    title(sprintf('greedy: %d/%d BFs, G_r = %.3g (%.2f dBi), G = %.3g, bound %.3g, Z_{in} = %.1f %+.1fj \\Omega', ...
        numel(Sg), N, Gr, 10*log10(Gr), G, Glb, real(Zin), imag(Zin)));
    exportgraphics(gcf, fullfile(thisDir, [tag '_greedyStructure.png']));

    results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', Ig, ...
        'part', 'abs', 'arrowScale', 'proportional');
    title(sprintf('|J| of the greedy structure, G_r = %.3g', Gr));
    exportgraphics(gcf, fullfile(thisDir, [tag '_greedyCurrent.png']));

    figure;
    plot(hist(:, 1), hist(:, 2), '.-', hist(:, 1), hist(:, 3), '.-'); hold on;
    yline(Glb, 'r--', sprintf('GCD bound %.3g', Glb));
    xlabel('flip'); ylabel('broadside gain'); grid on;
    legend({'(1-|\Gamma|^2)G', 'G'}, 'Location', 'southeast');
    exportgraphics(gcf, fullfile(thisDir, [tag '_greedyHistory.png']));
end

function [G, Zin] = portGain(I, port, Vp, Fth, Fph, Z0vac)
% G = 4 pi U/Pin, U = (|Fth I|^2 + |Fph I|^2)/(2 Z0), Pin = Re[Vp conj(I_port)]/2
U   = (abs(Fth*I)^2 + abs(Fph*I)^2)/(2*Z0vac);
G   = 4*pi*U/(0.5*real(Vp*conj(I(port))));
Zin = Vp/I(port);
end

function [Gr, G, Zin] = designGain(keep, Z, V, port, Vp, Fth, Fph, Z0vac, Zin0)
% realized gain (1 - |Gamma|^2) G of the structure with the basis functions keep
S = find(keep);
I = zeros(numel(V), 1);
I(S) = Z(S, S)\V(S);
[G, Zin] = portGain(I, port, Vp, Fth, Fph, Z0vac);
Gr = G*(1 - abs((Zin - Zin0)/(Zin + Zin0))^2);
end

function out = ifelse(cond, a, b)
if cond, out = a; else, out = b; end
end

function x = realizedCurrent(d, Lam, Zp, v)
% Lambda x = D (v - Z' x) -> x = (Lambda + D Z')^{-1} D v
x = (diag(Lam) + d.*Zp)\(d.*v);
end

function [r, Jac] = lagrangianResidual(dfree, free, port, Lam, Zp, v, xopt, C)
% r = [Re; Im] C (x(d) - xopt), dx/dd_j = M^{-1} e_j u_j
N = numel(v);
d = ones(N, 1); d(free) = dfree; d(port) = 1;
M = diag(Lam) + d.*Zp;
x = M\(d.*v);
e = C*(x - xopt);
r = [real(e); imag(e)];
if nargout > 1
    u   = v - Zp*x;
    CMi = C/M;
    Jc  = CMi(:, free).*u(free).';
    Jac = [real(Jc); imag(Jc)];
end
end
