%% Maximum scattered power of a copper 2:1 patch with local power and frequency-derivative balance
% Maximizes the scattered power Psca = I'*R0*I/2 of a copper structure inside
% the 2:1 patch (*_patch21ka1.txt: hexagonal 7 x 12 pixel grid, N = 233 RWG,
% ka = 1, f = 2.45 GHz, y-polarized plane wave along z, E0 = 1 kV/m).
%
% The dual bound with GCD is evaluated by VscatDerAugmented.py in the RWG basis
% with the non-diagonal material operator Z = Z0 + Zs L (main.tex, Sec.
% "Inference with Non-Diagonal Material Operator"), no Cholesky whitening. The
% variable is the scattered voltage augmented by its frequency derivative,
%   x = [Vsca; omega dVsca/domega],   Vsca = -Z0 I,
% and every basis function i of any structure satisfies the power and the
% frequency-derivative balance
%   conj(i_i) (V - Z I)_i = 0,
%   (V' - Z' I - Z I')_i = 0  (tested by conj(i_i) and conj(i'_i)),
% since the material does not change with frequency. The per-basis-function
% 2 x 2 tests are the projector constraints (Pstruct of 2 x 2 blocks).
%
% Derivatives: omega dZ0/domega from AToM's analytic 'ZmatD', omega dV/domega
% by a central difference of the plane wave in k, omega dZs/domega = Zs/2
% (skin effect, Zs ~ sqrt(omega)).
%
% Direct inference (Python): the density d_i of each basis function is the
% least-squares fit of the admittance relations of the current and its
% derivative,
%   Lam_ii i_i = d_i u_i,   Lam_ii (omega i'_i + i_i/2) = d_i omega u'_i,
% with Lam = Zs diag(L_ii), u = V - (Z - Lam) I, followed by a threshold sweep.
%
% stage = 'assemble' : mesh, operators, derivatives -> *_in.mat
% stage = 'dual'     : run VscatDerAugmented.py -> *_out.mat
% stage = 'post'     : *_out.mat -> table, figures
% stage = 'all'      : all of the above (default)
%
% Requires AToM (with the TopologySensitivity package) on the path and a
% Python environment with dolphindes (pythonExe).

if ~exist('stage', 'var')
    stage = 'post';
end
close all;

%% Settings
if ~exist('Ngcd', 'var')
    Ngcd = 20;              % number of GCD iterations
end
matrixSet = 'patch21ka1';
tag       = sprintf('VscatDerAug_%s_gcd%d', matrixSet, Ngcd);
f0        = 2.45e9;         % frequency (Hz)
sigma     = 5.96e7;         % copper conductivity (S/m)
E0        = 1e3;            % plane wave amplitude (V/m)
quadOrder = 12;             % reproduces X0_patch21ka1.txt to round-off
hFD       = 1e-4;           % relative step of the central differences

thisDir   = fileparts(mfilename('fullpath'));
resDir    = fullfile(thisDir, 'results');
if ~exist(resDir, 'dir'), mkdir(resDir); end
pythonExe = fullfile(thisDir, '..', '..', '.venv', 'bin', 'python');
pyScript  = fullfile(thisDir, 'VscatDerAugmented.py');
inFile    = fullfile(resDir, [tag '_in.mat']);
opFile    = fullfile(resDir, [tag '_operators.mat']);
if exist('variant', 'var') && ~isempty(variant)
    tag = [tag '_' variant];    % e.g. 'dPzero' (VscatDerAugmented.py with DPZERO=1)
end
outFile   = fullfile(resDir, [tag '_out.mat']);

%% Physical constants
c0    = models.utilities.constants.c0;
mu0   = models.utilities.constants.mu0;
omega = 2*pi*f0;
k0    = omega/c0;
delta = sqrt(2/(omega*mu0*sigma));    % skin depth
Zs    = (1 + 1i)/(sigma*delta);       % surface impedance, omega dZs/domega = Zs/2

%% Mesh (exportOmWpatch21ka1.m)
a = 1/k0;                                   % ka = 1
[~, ~, mesh] = models.utilities.meshPublic.pixelGridToHexaMesh(ones(7, 12), 1);
mesh.nodes = mesh.nodes./mesh.normDistanceA*a;
mesh = models.utilities.meshPublic.getMeshData2D(mesh.nodes, mesh.connectivityList);

if any(strcmp(stage, {'assemble', 'all'}))
    %% Operators (normalize = false, Ohm m^2) and omega dZ0/domega
    lmax = models.utilities.matrixOperators.MoM2D.SMatrix.lmax(k0, mesh.nodes);
    evalOP = @(f, req) models.utilities.matrixOperators.MoM2D.batch.evaluate(mesh, f, 0, ...
        'normalize', false, 'symmetrize', true, 'quadOrder', quadOrder, 'lmax', lmax, ...
        'requests', req);
    [OP, ~] = evalOP(f0, {'R0', 'X0', 'Lmat', 'ZmatD'});
    BF   = OP.BF;
    N    = BF.nUnknowns;
    sym  = @(M) (M + M.')/2;
    R0   = sym(real(OP.R0));
    X0   = sym(real(OP.X0));
    Lmat = sym(full(OP.Lmat));
    D    = sym(omega*OP.ZmatD);             % omega dZ0/domega (complex symmetric)

    %% Plane wave and omega dV/domega = k dV/dk (central difference)
    pw.propagationVector = [0 0 1];
    pw.initElectricField = [0 1 0];
    pw.axialRatio = Inf;
    pw.direction = 'right';
    Vk = models.solvers.MoM2D.excitation.planeWave(mesh, BF, pw, k0*[1 - hFD, 1, 1 + hFD], quadOrder);
    V  = Vk(:, 2);
    dV = (Vk(:, 3) - Vk(:, 1))/(2*hFD);

    %% Consistency checks
    relErr = @(A, B) norm(A - B, 'fro')/norm(B, 'fro');
    stored = struct('Lmat', Lmat, 'R0', R0, 'X0', X0, 'V', V);
    for name = fieldnames(stored).'
        ref = readmatrix(fullfile(thisDir, sprintf('%s_%s.txt', name{1}, matrixSet)));
        fprintf('%-4s: ||new - stored|| / ||stored|| = %.2e\n', name{1}, relErr(stored.(name{1}), ref));
    end
    [OPm, ~] = evalOP(f0*(1 - hFD), {'R0', 'X0'});
    [OPp, ~] = evalOP(f0*(1 + hFD), {'R0', 'X0'});
    Dfd = sym(((OPp.R0 - OPm.R0) + 1i*(OPp.X0 - OPm.X0))/(2*hFD));
    fprintf('omega dZ0/domega: ||ZmatD - FD|| / ||FD|| = %.2e (Re %.2e, Im %.2e)\n', relErr(D, Dfd), ...
        relErr(real(D), real(Dfd)), relErr(imag(D), imag(Dfd)));
    omWref = readmatrix(fullfile(thisDir, sprintf('omW_%s.txt', matrixSet)));
    fprintf('omega dX0/domega vs omW_%s.txt: %.2e\n', matrixSet, relErr(imag(D), omWref));
    fprintf('N = %d, Zs = %.4e%+.4ej Ohm, ||omega V''|| / ||V|| = %.3e, ||D|| / ||Z0|| = %.3f\n', ...
        N, real(Zs), imag(Zs), norm(dV)/norm(V), norm(D)/norm(R0 + 1i*X0));

    save(inFile, 'R0', 'X0', 'Lmat', 'D', 'V', 'dV', 'Zs', 'E0', 'Ngcd');
    save(opFile, 'mesh', 'BF', 'R0', 'X0', 'Lmat', 'D', 'V', 'dV', 'Zs', 'E0');
    fprintf('saved %s\n', inFile);
end

if any(strcmp(stage, {'dual', 'all'}))
    %% Dual bound with GCD and direct inference (Python)
    cmd = sprintf('"%s" -u "%s" "%s" "%s"', pythonExe, pyScript, inFile, outFile);
    fprintf('Running %s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('VscatDerAugmented.py failed (status %d).', status);
    end
end

if any(strcmp(stage, {'post', 'all'}))
    %% Bounds and inferred designs
    ops = load(opFile);
    out = load(outFile);
    BF  = ops.BF;
    fprintf('\n%-34s %12s\n', 'scattered power', 'W');
    rows = {'full plate', out.Pfull; ...
        'bound, power only, global', out.PlbBase; ...
        'bound, power only, GCD', out.PlbBaseGCD; ...
        'bound, power + derivative, global', out.PlbGlobal; ...
        'bound, power + derivative, GCD', out.PlbGCD; ...
        'bound, power + derivative, local', out.PlbLocal; ...
        'inferred d (LSQ), relaxed', out.Pd; ...
        'inferred d (power only), relaxed', out.PdPw; ...
        'inferred d (LSQ), best threshold', out.PsweepBest; ...
        'inferred d (power only), best thr.', out.PsweepBestPw};
    for iR = 1:size(rows, 1)
        fprintf('%-34s %12.6g\n', rows{iR, 1}, rows{iR, 2});
    end

    %% Figures
    % density per triangle (mean over its basis functions)
    nT = size(mesh.connectivityList, 1);
    triDens = @(d) accumarray([BF.data(:, 2); BF.data(:, 4)], [d; d], [nT 1]) ./ ...
        max(accumarray([BF.data(:, 2); BF.data(:, 4)], 1, [nT 1]), 1);
    dens = {out.d(:), out.dPw(:), out.dDer(:)};
    ttl  = {sprintf('LSQ, current + derivative, P = %.3g W', out.Pd), ...
        sprintf('current only, P = %.3g W', out.PdPw), ...
        sprintf('derivative only, P = %.3g W', out.PdDer)};
    figure('Position', [100 100 1200 400]);
    for iD = 1:3
        subplot(1, 3, iD);
        patch('Faces', mesh.connectivityList, 'Vertices', mesh.nodes, ...
            'FaceVertexCData', triDens(dens{iD}), 'FaceColor', 'flat', 'EdgeColor', 'none');
        axis equal off; colormap(flipud(gray)); clim([0 max(max(dens{iD}), eps)]); colorbar;
        title({['density, ' ttl{iD}], sprintf('max d = %.3g', max(dens{iD}))});
    end
    exportgraphics(gcf, fullfile(resDir, [tag '_inference.png']));

    S = find(out.Sbest(:));
    TSaux.plotStructure(mesh, BF, [], S, false);
    set(gcf, 'Name', sprintf('inferred: %d/%d BFs, P = %.3g W, bound %.3g W, full plate %.3g W', ...
        numel(S), BF.nUnknowns, out.PsweepBest, out.PlbGCD, out.Pfull), 'NumberTitle', 'off');
    exportgraphics(gcf, fullfile(resDir, [tag '_structure.png']));

    results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', out.Iopt(:), ...
        'part', 'abs', 'arrowScale', 'proportional');
    set(gcf, 'Name', sprintf('|J_{opt}| (GCD), P(I_{opt}) = %.3g W, bound %.3g W', out.PIopt, out.PlbGCD), 'NumberTitle', 'off');
    exportgraphics(gcf, fullfile(resDir, [tag '_dualCurrent.png']));

    results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', out.dIopt(:), ...
        'part', 'abs', 'arrowScale', 'proportional');
    set(gcf, 'Name', '|\omega dJ_{opt}/d\omega| (GCD)', 'NumberTitle', 'off');
    exportgraphics(gcf, fullfile(resDir, [tag '_dualCurrentDer.png']));

    results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', out.Ibest(:), ...
        'part', 'abs', 'arrowScale', 'proportional');
    set(gcf, 'Name', 'Inferred structure current', 'NumberTitle', 'off');
    exportgraphics(gcf, fullfile(resDir, [tag '_structureCurrent.png']));

    figure;
    semilogy(out.sweepN(:), out.sweepP(:), '.-', out.sweepN(:), out.sweepPpw(:), '.-'); hold on;
    yline(out.PlbGCD, 'r--', sprintf('power + derivative GCD %.4g', out.PlbGCD), ...
        'LabelVerticalAlignment', 'top');
    yline(out.PlbBaseGCD, 'b-.', sprintf('power-only GCD %.4g', out.PlbBaseGCD), ...
        'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'left');
    yline(out.Pfull, 'g-', sprintf('full plate %.3g', out.Pfull));
    xlabel('number of metal BFs (sorted by density)'); ylabel('scattered power (W)'); grid on;
    legend({'LSQ, current + derivative', 'current only'}, 'Location', 'southeast');
    exportgraphics(gcf, fullfile(resDir, [tag '_sweep.png']));
end
