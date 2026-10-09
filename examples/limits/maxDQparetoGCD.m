%% Directivity-Q Pareto bound of a PEC 2:1 plate fed by a 50 Ohm port, dual bound with GCD
% Bounds the broadside directivity D = 4*pi*U/P (theta = 0, both
% polarizations) against the quality factor Q of a PEC structure inside a
% 2:1 rectangle, ka = 1, f = 2.45 GHz, fed by a delta gap at the basis
% function nearest the midpoint of the longer edge (setup of minQportGCD.m).
% The input voltage 1 V and the input impedance 50 Ohm fix the input current
% I0 = 20 mA and the radiated power P = 10 mW.
%
% The maximum directivity of a PEC structure is unbounded, the stored energy
% regularizes it. maxDQparetoGCD.py evaluates, for a sweep of weights nu, the
% dual bound g(nu) >= max (D - nu Q) with GCD refinement in the
% scattered-voltage variable Vsca = -Z0*I (as minQportGCD.py). Every structure
% matched to Zin0 satisfies D <= g(nu) + nu Q, the Pareto bound is the envelope
%   D_ub(Q) = min_nu [g(nu) + nu Q].
% Designs are inferred from the dual optimal current of every nu as in
% minQportGCD.m, z_n = (V + Vsca)_n/I_n (metal |z| -> 0, void |z| -> Inf), by a
% threshold sweep over |z| (port always kept).
%
% stage = 'assemble' : mesh, normalized operators, port -> *_in.mat
% stage = 'dual'     : run maxDQparetoGCD.py -> *_out.mat
% stage = 'post'     : *_out.mat -> Pareto envelope, inferred designs, figures
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
if ~exist('nNu', 'var')
    nNu = 13;               % number of weights nu
end
if ~exist('nuLog', 'var')
    nuLog = [-3 1];         % nu = nuMax*10.^linspace(nuLog(1), nuLog(2), nNu)
end
tag = sprintf('maxDQpareto_patch21ka1_gcd%d', Ngcd);
if m ~= 25
    tag = sprintf('%s_m%d', tag, m);
end
ka          = 1;
fixMeshToKa = 2;
f0          = 2.45e9;       % frequency (Hz)
Vp          = 1;            % input voltage (V)
Zin0        = 50;           % required input impedance (Ohm)
I0          = Vp/Zin0;      % input current (A)
thetaDir    = 0;            % directivity direction (broadside)
phiDir      = 0;
quadOrder   = 7;
GammaMax    = 1/3;          % matched designs |Gamma| < GammaMax

thisDir   = fileparts(mfilename('fullpath'));
pythonExe = fullfile(thisDir, '..', '..', '.venv', 'bin', 'python');
pyScript  = fullfile(thisDir, 'maxDQparetoGCD.py');
inFile    = fullfile(thisDir, [tag '_in.mat']);
outFile   = fullfile(thisDir, [tag '_out.mat']);
opFile    = fullfile(thisDir, [tag '_operators.mat']);

%% Physical constants
c0    = models.utilities.constants.c0;
Z0vac = models.utilities.constants.Z0;
k0    = 2*pi*f0/c0;
lam   = c0/f0;

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
    %% Operators (normalized -> Ohm, port voltage in V, terminal current in A)
    lmax = models.utilities.matrixOperators.MoM2D.SMatrix.lmax(k0, mesh.nodes);
    [OP, ~] = models.utilities.matrixOperators.MoM2D.batch.evaluate(mesh, f0, 0, ...
        'normalize', true, 'symmetrize', true, 'quadOrder', quadOrder, ...
        'lmax', lmax, 'requests', {'R0', 'X0', 'omW'});
    BF  = OP.BF;
    N   = BF.nUnknowns;
    R0  = real(OP.R0);  R0  = (R0 + R0.')/2;
    X0  = real(OP.X0);  X0  = (X0 + X0.')/2;
    omW = real(OP.omW); omW = (omW + omW.')/2;
    [~, Fph, Fth] = models.utilities.matrixOperators.MoM2D.farfield.computeU( ...
        mesh, BF, f0, thetaDir, phiDir, 'total', quadOrder);
    dn  = 1./mesh.triangleEdgeLengths(BF.data(:, 3));   % normalization 1/l_n
    Fth = Fth(:).'.*dn.';
    Fph = Fph(:).'.*dn.';

    %% Port: BF nearest the midpoint of the longer (bottom, y = -wid/2) edge
    feederPosition = [0, min(mesh.nodes(:, 2)) + h/2, 0];
    edgeCenters = mesh.triangleEdgeCenters(BF.data(:, 3), :);
    [dist, port] = min(models.utilities.vectorOperations.rowNorm(edgeCenters - feederPosition));
    fprintf('N = %d, size %.4f x %.4f m, port BF %d at [%.4f %.4f] (%.1e m)\n', ...
        N, len, wid, port, edgeCenters(port, 1:2), dist);

    % full plate fed at the port (reference)
    V = zeros(N, 1); V(port) = Vp;
    Ifull = (R0 + 1i*X0)\V;
    [Dfull, Qfull, ZinFull] = portDQ(Ifull, port, Vp, R0, X0, omW, Fth, Fph, Z0vac);
    fprintf('full plate: Zin = %.3f %+.3fj Ohm, D = %.4g, Q = %.4g\n', ...
        real(ZinFull), imag(ZinFull), Dfull, Qfull);

    port = port - 1; %#ok<NASGU> zero based in Python
    save(inFile, 'R0', 'X0', 'omW', 'Fth', 'Fph', 'port', 'I0', 'Vp', 'Ngcd');
    port = port + 1;
    save(opFile, 'mesh', 'BF', 'R0', 'X0', 'omW', 'Fth', 'Fph', 'port', ...
        'Dfull', 'Qfull', 'ZinFull');
    fprintf('saved %s\n', inFile);
end

if any(strcmp(stage, {'dual', 'all'}))
    %% Dual bounds for the nu sweep with GCD (Python)
    cmd = sprintf('NU=%d NULOG=%g,%g "%s" -u "%s" "%s" "%s"', nNu, nuLog, ...
        pythonExe, pyScript, inFile, outFile);
    fprintf('Running %s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('maxDQparetoGCD.py failed (status %d).', status);
    end
end

if any(strcmp(stage, {'post', 'all'}))
    %% Dual optima
    ops = load(opFile);
    out = load(outFile);
    BF = ops.BF; R0 = ops.R0; X0 = ops.X0; omW = ops.omW; port = ops.port;
    Fth = ops.Fth; Fph = ops.Fph;
    N  = BF.nUnknowns;
    Z0 = R0 + 1i*X0;
    V  = zeros(N, 1); V(port) = Vp;
    nu = out.nu(:).'; gGl = out.gGlobal(:).'; gGCD = out.gGCD(:).';
    nNu = numel(nu);

    % Pareto envelope D_ub(Q) = min_nu [g(nu) + nu Q]
    Qmin = 0.9*min(out.Qopt);
    Qgrid = logspace(log10(Qmin), log10(2*max(out.Qopt)), 400);
    DubGl  = min(gGl.' + nu.'.*Qgrid, [], 1);
    DubGCD = min(gGCD.' + nu.'.*Qgrid, [], 1);
    QlbMin = NaN;           % minimum Q bound (minQportGCD)
    qFile = fullfile(thisDir, 'minQport_patch21ka1_out.mat');
    if m == 25 && isfile(qFile)
        q = load(qFile, 'QlbGCD');
        QlbMin = q.QlbGCD;
    end

    %% Inference for every nu: threshold sweep over |z|, z = (V + Vsca)./Iopt
    nList = 2:N;
    dsg = struct('nu', num2cell(nu), 'order', [], 'D', [], 'Q', [], 'Zin', [], 'Gamma', []);
    for iNu = 1:nNu
        Iopt = out.Iopt(:, iNu);
        Vsca = out.Vsca(:, iNu);
        z = (V + Vsca)./Iopt;
        z(port) = 0;
        [~, order] = sort(abs(z));
        [D, Q, Gam] = deal(nan(size(nList)));
        Zin = nan(size(nList));
        for iN = 1:numel(nList)
            S = sort(order(1:nList(iN)));
            I = zeros(N, 1);
            I(S) = Z0(S, S)\V(S);
            [D(iN), Q(iN), Zin(iN)] = portDQ(I, port, Vp, R0, X0, omW, Fth, Fph, Z0vac);
            Gam(iN) = abs((Zin(iN) - Zin0)/(Zin(iN) + Zin0));
        end
        dsg(iNu).order = order; dsg(iNu).D = D; dsg(iNu).Q = Q;
        dsg(iNu).Zin = Zin; dsg(iNu).Gamma = Gam;
        fprintf(['nu = %.4g: g_global = %.4g, g_GCD = %.4g, Iopt D = %.3g Q = %.3g, ' ...
            'inferred max(D - nu Q) = %.4g (matched %.4g)\n'], nu(iNu), gGl(iNu), gGCD(iNu), ...
            out.Dopt(iNu), out.Qopt(iNu), max(D - nu(iNu)*Q), ...
            max([-Inf, D(Gam < GammaMax) - nu(iNu)*Q(Gam < GammaMax)]));
    end

    % all inferred designs, non-dominated (min Q, max D) among the matched ones
    allD = [dsg.D]; allQ = [dsg.Q]; allG = [dsg.Gamma];
    allNu = repelem(1:nNu, numel(nList)); allN = repmat(nList, 1, nNu);
    matched = allG < GammaMax;
    mLabel = sprintf('|\\Gamma| < %.2g', GammaMax);
    if ~any(matched)
        mLabel = '(none matched)';
        warning('no inferred design with |Gamma| < %.2f, Pareto set over all designs', GammaMax);
        matched = true(size(allG));
    end
    idx = find(matched);
    [~, srt] = sortrows([allQ(idx).', -allD(idx).']);
    idx = idx(srt);
    pareto = idx(1);
    for k = idx(2:end)
        if allD(k) > allD(pareto(end))
            pareto(end + 1) = k; %#ok<AGROW>
        end
    end
    fprintf('Pareto set of matched inferred designs:\n');
    for k = pareto
        fprintf('  nu #%2d, %3d BFs: Q = %7.3f, D = %.4f, Zin = %6.2f %+7.2fj, D_ub(Q) = %.4f\n', ...
            allNu(k), allN(k), allQ(k), allD(k), real(dsg(allNu(k)).Zin(allN(k) - 1)), ...
            imag(dsg(allNu(k)).Zin(allN(k) - 1)), min(gGCD + nu*allQ(k)));
    end
    viol = allD(matched) - min(gGCD.' + nu.'.*allQ(matched), [], 1);
    fprintf('max (D - D_ub(Q)) over matched inferred designs: %.3e\n', max(viol));

    save(fullfile(thisDir, [tag '_post.mat']), 'dsg', 'Qgrid', 'DubGl', 'DubGCD', ...
        'allD', 'allQ', 'allG', 'allNu', 'allN', 'pareto', 'QlbMin');

    %% Figures
    figure('Position', [100 100 760 520]);
    hA = semilogx(allQ, allD, '.', 'Color', 0.75*[1 1 1], 'MarkerSize', 5); hold on;
    hM = semilogx(allQ(matched), allD(matched), '.', 'Color', [0.45 0.6 0.85], 'MarkerSize', 6);
    hP = semilogx(allQ(pareto), allD(pareto), 'ko-', 'MarkerFaceColor', 'y', 'MarkerSize', 5);
    hG = semilogx(Qgrid, DubGl, 'k:', 'LineWidth', 1.2);
    hC = semilogx(Qgrid, DubGCD, 'r-', 'LineWidth', 1.8);
    hO = semilogx(out.Qopt, out.Dopt, 'r^', 'MarkerSize', 6);
    hF = semilogx(ops.Qfull, ops.Dfull, 'gs', 'MarkerFaceColor', 'g', 'MarkerSize', 8);
    hh = [hA hM hP hG hC hO hF];
    lg = {'inferred designs', mLabel, 'Pareto set (inferred)', ...
        'global bound', sprintf('GCD bound (%d iter.)', Ngcd), 'dual optimal currents', 'full plate'};
    if isfinite(QlbMin)
        hh(end + 1) = xline(QlbMin, 'b--', 'LineWidth', 1.2);
        lg{end + 1} = sprintf('min Q bound %.3g', QlbMin);
    end
    yl = ylim; ylim([0 min(yl(2), 1.3*max(DubGCD(Qgrid <= 2*max(allQ(pareto)))))]);
    xlim([0.9*min([allQ(pareto), Qmin]), 2*max(out.Qopt)]);
    xlabel('Q'); ylabel('broadside directivity D'); grid on;
    legend(hh, lg, 'Location', 'southeast');
    title('D-Q Pareto front, PEC 2:1 plate, ka = 1, Z_{in} = 50 \Omega');
    exportgraphics(gcf, fullfile(thisDir, [tag '_pareto.png']));

    figure;
    semilogx(nu, gGl, 'k.:', nu, gGCD, 'r.-'); hold on;
    xline(out.nuMax, 'b--', '\nu_{max}');
    xlabel('\nu'); ylabel('g(\nu) \geq max (D - \nu Q)'); grid on;
    legend('global', 'GCD', 'Location', 'best');
    exportgraphics(gcf, fullfile(thisDir, [tag '_gnu.png']));

    % representative designs: min Q, knee (max distance to the chord), max D
    Pq = allQ(pareto); Pd = allD(pareto);
    lq = log10(Pq);
    chord = abs((Pd(end) - Pd(1))*(lq - lq(1)) - (lq(end) - lq(1))*(Pd - Pd(1)));
    [~, iKnee] = max(chord);
    pick = unique([1, iKnee, numel(pareto)], 'stable');
    names = {'lowQ', 'knee', 'highD'};
    if numel(pick) < 3, names = names([1 3]); end
    th = linspace(-pi/2, pi/2, 61);
    planes = [0 pi/2];
    dn = 1./mesh.triangleEdgeLengths(BF.data(:, 3));
    for iP = 1:numel(pick)
        k = pareto(pick(iP));
        iNu = allNu(k); nB = allN(k);
        S = sort(dsg(iNu).order(1:nB));
        I = zeros(N, 1); I(S) = Z0(S, S)\V(S);
        [Dk, Qk, Zk] = portDQ(I, port, Vp, R0, X0, omW, Fth, Fph, Z0vac);
        dTag = sprintf('%s_%s', tag, names{iP});
        fprintf('%-5s: nu #%d, %d/%d BFs, Q = %.3f, D = %.4f, Zin = %.2f %+.2fj, D_ub(Q) = %.4f\n', ...
            names{iP}, iNu, numel(S), N, Qk, Dk, real(Zk), imag(Zk), min(gGCD + nu*Qk));

        TSaux.plotStructure(mesh, BF, port, S, false);
        title(sprintf('%s: %d/%d BFs, Q = %.3g, D = %.3g (bound %.3g), Z_{in} = %.1f %+.1fj \\Omega', ...
            names{iP}, numel(S), N, Qk, Dk, min(gGCD + nu*Qk), real(Zk), imag(Zk)));
        exportgraphics(gcf, fullfile(thisDir, [dTag '_structure.png']));

        results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', I, ...
            'part', 'abs', 'arrowScale', 'proportional');
        title(sprintf('|J| of the %s design, Q = %.3g, D = %.3g', names{iP}, Qk, Dk));
        exportgraphics(gcf, fullfile(thisDir, [dTag '_current.png']));

        % directivity pattern in the xz (phi = 0) and yz (phi = 90 deg) planes
        figure;
        for iPl = 1:2
            Dcut = zeros(size(th));
            for it = 1:numel(th)
                [~, fph, fth] = models.utilities.matrixOperators.MoM2D.farfield.computeU( ...
                    mesh, BF, f0, abs(th(it)), planes(iPl) + pi*(th(it) < 0), 'total', quadOrder);
                Dcut(it) = portDQ(I, port, Vp, R0, X0, omW, fth(:).'.*dn.', fph(:).'.*dn.', Z0vac);
            end
            polarplot(th, Dcut, 'LineWidth', 1.5); hold on;
        end
        pax = gca; pax.ThetaZeroLocation = 'top'; pax.ThetaDir = 'clockwise';
        pax.ThetaLim = [-90 90];
        legend({'xz plane', 'yz plane'}, 'Location', 'southoutside');
        title(sprintf('directivity of the %s design', names{iP}));
        exportgraphics(gcf, fullfile(thisDir, [dTag '_pattern.png']));
    end
end

function [D, Q, Zin] = portDQ(I, port, Vp, R0, X0, omW, Fth, Fph, Z0vac)
% D = 4 pi U/P, U = (|Fth I|^2 + |Fph I|^2)/(2 Z0), P = I'R0 I/2
% Q = (I'omW I + |I'X0 I|)/(2 I'R0 I) (minQselfRes), Zin = Vp/I_port
P   = 0.5*real(I'*R0*I);
D   = 4*pi*(abs(Fth*I)^2 + abs(Fph*I)^2)/(2*Z0vac)/P;
Q   = 0.5*real(I'*omW*I + abs(I'*X0*I))/real(I'*R0*I);
Zin = Vp/I(port);
end
