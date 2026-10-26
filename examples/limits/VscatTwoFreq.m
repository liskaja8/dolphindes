%% Maximum scattered power of a copper 2:1 patch at two nearby frequencies
% Maximizes the mean scattered power (P+ + P-)/2, P = I'*R0*I/2, of one copper
% structure inside the 2:1 patch (*_patch21ka1.txt: hexagonal 7 x 12 pixel
% grid, N = 233 RWG, ka = 1 at f0 = 2.45 GHz, y-polarized plane wave along z,
% E0 = 1 kV/m) at the frequencies f+- = f0 (1 +- h).
%
% The dual bound with GCD is evaluated by VscatTwoFreq.py in the RWG basis with
% Z = Z0 + Zs L (main.tex, Sec. "Frequency-Derivative Power Balance"), two-
% frequency variant: the variable is the pair of scattered voltages
%   x = [Vsca+; Vsca-],   Vsca+- = -Z0+- I+-,
% so that the frequency derivative omega I' = (I+ - I-)/(2h) is defined by
% construction. Every basis function i of a structure satisfies
%   conj(i+_i) (V+ - Z+ I+)_i = 0,   conj(i-_i) (V- - Z- I-)_i = 0,
% (power balance at each frequency, no cross terms) and the derivative balance
% (center-frequency operators, exact to O(h^2))
%   (Z/(2h) + Z_w/2) I+ - (Z/(2h) - Z_w/2) I- = V_w,   Z_w = omega dZ/domega,
% tested by conj(i+ + i-) and conj(i+ - i-).
%
% Derivatives: omega dZ0/domega from AToM's analytic 'ZmatD', omega dV/domega
% by a central difference of the plane wave in k, omega dZs/domega = Zs/2.
%
% stage = 'assemble' : mesh, operators at f0, f0 (1 +- h) -> *_in.mat
% stage = 'dual'     : run VscatTwoFreq.py -> *_out.mat
% stage = 'post'     : *_out.mat -> table, figures
% stage = 'all'      : all of the above (default)
%
% Requires AToM (with the TopologySensitivity package) on the path, or the
% environment variable ATOM_DIR (cluster), and a Python environment with
% dolphindes (pythonExe; on the cluster the script is run by
% narval_VscatTwoFreq.sh instead of stage 'dual').

if ~exist('stage', 'var')
    stage = 'all';
end
close all;

atomDir = getenv('ATOM_DIR');
if ~isempty(atomDir)
    addpath(atomDir);
    addpath(fullfile(atomDir, 'Packages', 'TopologySensitivity'));
end

%% Settings
if ~exist('Ngcd', 'var')
    Ngcd = str2double(getenv('NGCD'));
    if isnan(Ngcd), Ngcd = 20; end  % number of GCD iterations
end
matrixSet = 'patch21ka1';
f0        = 2.45e9;         % center frequency (Hz)
h         = 1e-4;           % relative frequency step, f+- = f0 (1 +- h)
sigma     = 5.96e7;         % copper conductivity (S/m)
E0        = 1e3;            % plane wave amplitude (V/m)
quadOrder = 12;             % reproduces X0_patch21ka1.txt to round-off
tagIn     = sprintf('VscatTwoFreq_%s_h%.0e', matrixSet, h);
tag       = sprintf('%s_gcd%d', tagIn, Ngcd);

thisDir   = fileparts(mfilename('fullpath'));
resDir    = fullfile(thisDir, 'results');
if ~exist(resDir, 'dir'), mkdir(resDir); end
pythonExe = fullfile(thisDir, '..', '..', '.venv', 'bin', 'python');
pyScript  = fullfile(thisDir, 'VscatTwoFreq.py');
inFile    = fullfile(resDir, [tagIn '_in.mat']);
opFile    = fullfile(resDir, [tagIn '_operators.mat']);
if exist('variant', 'var') && ~isempty(variant)
    tag = [tag '_' variant];
end
outFile   = fullfile(resDir, [tag '_out.mat']);

%% Physical constants
c0    = models.utilities.constants.c0;
mu0   = models.utilities.constants.mu0;
omega = 2*pi*f0;
k0    = omega/c0;
delta = sqrt(2/(omega*mu0*sigma));    % skin depth at f0
Zs    = (1 + 1i)/(sigma*delta);       % surface impedance, Zs ~ sqrt(omega)

%% Mesh (exportOmWpatch21ka1.m), fixed at ka = 1 for f0
a = 1/k0;
[~, ~, mesh] = models.utilities.meshPublic.pixelGridToHexaMesh(ones(7, 12), 1);
mesh.nodes = mesh.nodes./mesh.normDistanceA*a;
mesh = models.utilities.meshPublic.getMeshData2D(mesh.nodes, mesh.connectivityList);

if any(strcmp(stage, {'assemble', 'all'}))
    %% Operators (normalize = false, Ohm m^2) at f0 and f0 (1 +- h)
    lmax = models.utilities.matrixOperators.MoM2D.SMatrix.lmax(k0*(1 + h), mesh.nodes);
    evalOP = @(f, req) models.utilities.matrixOperators.MoM2D.batch.evaluate(mesh, f, 0, ...
        'normalize', false, 'symmetrize', true, 'quadOrder', quadOrder, 'lmax', lmax, ...
        'requests', req);
    sym  = @(M) (M + M.')/2;
    [OP, ~]  = evalOP(f0, {'R0', 'X0', 'Lmat', 'ZmatD'});
    [OPm, ~] = evalOP(f0*(1 - h), {'R0', 'X0'});
    [OPp, ~] = evalOP(f0*(1 + h), {'R0', 'X0'});
    BF   = OP.BF;
    N    = BF.nUnknowns;
    R0   = sym(real(OP.R0));
    X0   = sym(real(OP.X0));
    Lmat = sym(full(OP.Lmat));
    D    = sym(omega*OP.ZmatD);             % omega dZ0/domega (complex symmetric)
    R0p  = sym(real(OPp.R0));  X0p = sym(real(OPp.X0));
    R0m  = sym(real(OPm.R0));  X0m = sym(real(OPm.X0));
    Zsp  = Zs*sqrt(1 + h);     Zsm = Zs*sqrt(1 - h);

    %% Plane wave at f0 (1 - h), f0, f0 (1 + h), omega dV/domega (central difference)
    pw.propagationVector = [0 0 1];
    pw.initElectricField = [0 1 0];
    pw.axialRatio = Inf;
    pw.direction = 'right';
    Vk = models.solvers.MoM2D.excitation.planeWave(mesh, BF, pw, k0*[1 - h, 1, 1 + h], quadOrder);
    Vm = Vk(:, 1);
    V  = Vk(:, 2);
    Vp = Vk(:, 3);
    dV = (Vp - Vm)/(2*h);

    %% Consistency checks
    relErr = @(A, B) norm(A - B, 'fro')/norm(B, 'fro');
    stored = struct('Lmat', Lmat, 'R0', R0, 'X0', X0, 'V', V);
    for name = fieldnames(stored).'
        ref = readmatrix(fullfile(thisDir, sprintf('%s_%s.txt', name{1}, matrixSet)));
        fprintf('%-4s: ||new - stored|| / ||stored|| = %.2e\n', name{1}, relErr(stored.(name{1}), ref));
    end
    Dfd = sym(((R0p - R0m) + 1i*(X0p - X0m))/(2*h));
    fprintf('omega dZ0/domega: ||ZmatD - FD|| / ||FD|| = %.2e\n', relErr(D, Dfd));
    fprintf('N = %d, Zs = %.4e%+.4ej Ohm, ||Z0+ - Z0-|| / ||Z0|| = %.3e, ||omega V''|| / ||V|| = %.3e\n', ...
        N, real(Zs), imag(Zs), norm((R0p - R0m) + 1i*(X0p - X0m), 'fro')/norm(R0 + 1i*X0, 'fro'), ...
        norm(dV)/norm(V));

    save(inFile, 'R0', 'X0', 'Lmat', 'D', 'V', 'dV', 'Zs', 'R0p', 'X0p', 'Vp', 'Zsp', ...
        'R0m', 'X0m', 'Vm', 'Zsm', 'h', 'E0', 'Ngcd');
    save(opFile, 'mesh', 'BF', 'R0', 'X0', 'Lmat', 'D', 'V', 'dV', 'Zs', 'R0p', 'X0p', 'Vp', 'Zsp', ...
        'R0m', 'X0m', 'Vm', 'Zsm', 'h', 'E0');
    fprintf('saved %s\n', inFile);
end

if any(strcmp(stage, {'dual', 'all'}))
    %% Dual bound with GCD and direct inference (Python)
    cmd = sprintf('NGCD=%d "%s" -u "%s" "%s" "%s"', Ngcd, pythonExe, pyScript, inFile, outFile);
    fprintf('Running %s\n', cmd);
    status = system(cmd);
    if status ~= 0
        error('VscatTwoFreq.py failed (status %d).', status);
    end
end

if any(strcmp(stage, {'post', 'all'}))
    %% Bounds and inferred designs
    ops = load(opFile);
    out = load(outFile);
    BF  = ops.BF;
    fprintf('\n%-38s %12s\n', 'mean scattered power (P+ + P-)/2', 'W');
    rows = {'full plate', out.Pfull; ...
        'bound, power only f+, GCD', out.PlbBasePlus; ...
        'bound, power only f-, GCD', out.PlbBaseMinus; ...
        'bound, two frequencies, global', out.PlbGlobal; ...
        'bound, two frequencies, GCD', out.PlbGCD; ...
        'dual current (shifted), mean P', out.PIopt; ...
        'inferred d, relaxed', out.Pd; ...
        'inferred d, best threshold', out.PsweepBest};
    for iR = 1:size(rows, 1)
        fprintf('%-38s %12.6g\n', rows{iR, 1}, rows{iR, 2});
    end
    fprintf('||(I+ - I-)/(2h)|| / ||I|| = %.3e (dual, shifted), %.3e (inferred structure)\n', ...
        out.dIratio, out.dIratioBest);

    %% Figures
    nT = size(mesh.connectivityList, 1);
    triDens = @(d) accumarray([BF.data(:, 2); BF.data(:, 4)], [d; d], [nT 1]) ./ ...
        max(accumarray([BF.data(:, 2); BF.data(:, 4)], 1, [nT 1]), 1);
    figure;
    patch('Faces', mesh.connectivityList, 'Vertices', mesh.nodes, ...
        'FaceVertexCData', triDens(out.d(:)), 'FaceColor', 'flat', 'EdgeColor', 'none');
    axis equal off; colormap(flipud(gray)); clim([0 max(max(out.d(:)), eps)]); colorbar;
    title({sprintf('density (LSQ, f+ and f-), P = %.3g W', out.Pd), sprintf('max d = %.3g', max(out.d(:)))});
    exportgraphics(gcf, fullfile(resDir, [tag '_inference.png']));

    S = find(out.Sbest(:));
    TSaux.plotStructure(mesh, BF, [], S, false);
    set(gcf, 'Name', sprintf('inferred: %d/%d BFs, P = %.3g W, bound %.3g W, full plate %.3g W', ...
        numel(S), BF.nUnknowns, out.PsweepBest, out.PlbGCD, out.Pfull), 'NumberTitle', 'off');
    exportgraphics(gcf, fullfile(resDir, [tag '_structure.png']));

    results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', out.Iplus(:), ...
        'part', 'abs', 'arrowScale', 'proportional');
    set(gcf, 'Name', sprintf('|J_+| (GCD, shifted), mean P = %.3g W, bound %.3g W', out.PIopt, out.PlbGCD), 'NumberTitle', 'off');
    exportgraphics(gcf, fullfile(resDir, [tag '_dualCurrent.png']));

    results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', out.dIfd(:), ...
        'part', 'abs', 'arrowScale', 'proportional');
    set(gcf, 'Name', '|(J_+ - J_-)/(2h)| (GCD, shifted)', 'NumberTitle', 'off');
    exportgraphics(gcf, fullfile(resDir, [tag '_dualCurrentDer.png']));

    figure;
    semilogy(out.sweepN(:), out.sweepP(:), '.-'); hold on;
    yline(out.PlbGCD, 'r--', sprintf('two-frequency GCD %.4g', out.PlbGCD), 'LabelVerticalAlignment', 'top');
    yline(out.Pfull, 'g-', sprintf('full plate %.3g', out.Pfull));
    xlabel('number of metal BFs (sorted by density)'); ylabel('mean scattered power (W)'); grid on;
    exportgraphics(gcf, fullfile(resDir, [tag '_sweep.png']));
end
