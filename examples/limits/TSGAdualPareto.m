%% TSGA Pareto front: scattered power vs. distance from the dual optimum
% Maximizes the scattered power normalized by the power incident on the
% design domain, Psca/Pinc. The dual bound with GCD refinement is evaluated
% by dualGCDforMTB.py in the scattered-voltage variable Vsca = -Z0*I. Its
% dual optimal Vopt and Lagrange matrix LV define the second objective
%     J = (Vsca - Vopt)' * LV * (Vsca - Vopt),
% and NSGA-II + topology sensitivity (TSGA) builds the Pareto front
% [max Psca/Pinc, min J].
%
% Requires AToM (with the TopologySensitivity package and FOPS) on the path
% and a Python environment with dolphindes (pythonExe).

clc;
clear;
close all;

%% Settings
structure       = 'patch21ka1'; % 'patch21ka1' (2:1 plate, EuCAP27 ortho mesh) | 'dipole' (hexa strip)
loadPrecomputed = true;     % load operators cached by a previous run (same mesh)
loadFromTxt     = false;    % dipole only: load R0/X0/Lmat/V_dipole.txt (match the hexa mesh)

% 2:1 plate (EuCAP27/code/getMatricesFixedMesh.m): half diagonal a = ka/k0,
% Ny x 2Ny pixels with four triangles, Ny from m basis functions per wavelength
% at the fixed-mesh size ka = fixMeshToKa
ka          = 1;
m           = 50;          % EuCAP27 default 25 (N = 285), 50 -> N = 744
fixMeshToKa = 2;
polarization = 'x';         % 'x' along the longer side | 'y' along the shorter side (txt data)

f0    = 2.45e9;             % frequency (Hz)
sigma = 5.96e7;             % copper conductivity (S/m)
E0    = 1;                  % plane wave amplitude (V/m), Psca/Pinc is independent of E0
Ngcd  = 20;                 % number of GCD iterations

thisDir   = fileparts(mfilename('fullpath'));
pythonExe = fullfile(thisDir, '..', '..', '.venv', 'bin', 'python');
pyScript  = fullfile(thisDir, 'dualGCDforMTB.py');

% TSGA (NSGA-II + topology sensitivity)
nAgents = 32;
nIters  = 30;
useParallel = false; % the NSGA solver broadcasts its progress figure to every
                     % parfor worker, memory explodes with a pool; serial is fast

%% Physical constants
c0    = models.utilities.constants.c0;
Z0vac = models.utilities.constants.Z0;
mu0   = models.utilities.constants.mu0;
k0    = 2*pi*f0/c0;
lam   = c0/f0;
omega = 2*pi*f0;
delta = sqrt(2/(omega*mu0*sigma));    % skin depth
Zs    = (1 + 1i)/(sigma*delta);       % surface impedance (as in dolphindes notebooks)

%% Mesh
switch structure
    case 'patch21ka1'
        a     = ka/k0;
        len   = 4/sqrt(5)*a;
        wid   = len/2;
        mEff  = m*fixMeshToKa/ka;
        Ny    = ceil((wid/lam*mEff + 1)/2);
        [~, ~, meshPx] = models.utilities.meshPublic.pixelGridToOrthoMesh(2*ones(Ny, 2*Ny), 1);
        nodes = meshPx.nodes./meshPx.normDistanceA*sqrt((wid/2)^2 + (len/2)^2);
        conn  = meshPx.connectivityList;
        Epol  = double(polarization == ['x' 'y' 'z']);
    case 'dipole'
        % shortened half-wavelength strip, width 120 times shorter than the
        % original length, one row of equilateral triangles along y
        % (Montreal/codes/dipoleMeshHexa.m, operators of *_dipole.txt)
        len   = 0.94*lam/2;
        wid   = lam/2/120;
        e2h   = sqrt(3)/2;
        Nx    = round(len/wid*e2h);
        [nodes, conn] = models.utilities.meshPublic.pixelGridToHexaMesh(ones(1, Nx), wid/e2h);
        nodes = nodes(:, [2 1 3]);
        Epol  = [0 1 0];   % along the dipole
    otherwise
        error('Unknown structure "%s".', structure);
end
mesh = models.utilities.meshPublic.getMeshData2D(nodes, conn);
BF   = models.solvers.MoM2D.basisFcns.getBasisFcns(mesh);
N    = BF.nUnknowns;

%% Operators (AToM, normalize = false, unit plane wave)
tag = structure;
if strcmp(structure, 'patch21ka1')
    tag = sprintf('%s_%s_m%d', structure, polarization, m);
end
cacheFile = fullfile(thisDir, ['TSGAdualPareto_' tag '_operators.mat']);
useCache  = false;
if loadPrecomputed && isfile(cacheFile)
    cache    = load(cacheFile);
    useCache = isfield(cache, 'Epol') && isequal(cache.Epol, Epol) && ...
        isequal(size(cache.mesh.nodes), size(mesh.nodes)) && ...
        norm(cache.mesh.nodes - mesh.nodes, 'fro') <= 1e-12*norm(mesh.nodes, 'fro');
end
if loadFromTxt
    if ~strcmp(structure, 'dipole')
        error('Operators of "%s" are not stored in txt files.', structure);
    end
    fprintf('Loading operators from %s/*_%s.txt\n', thisDir, structure);
    readOp = @(name) readmatrix(fullfile(thisDir, [name '_' structure '.txt']));
    R0   = readOp('R0');
    X0   = readOp('X0');
    Lmat = readOp('Lmat');
    V    = readOp('V');
    V    = V(:);
    Lchk = models.utilities.matrixOperators.MoM2D.ohmicLosses.computeL(mesh, BF);
    if numel(V) ~= N || norm(Lchk - Lmat, 'fro') > 1e-6*norm(Lmat, 'fro')
        error('The txt operators do not match the %s mesh.', structure);
    end
elseif useCache
    fprintf('Loading precomputed operators from %s\n', cacheFile);
    R0   = cache.R0;
    X0   = cache.X0;
    Lmat = cache.Lmat;
    V    = cache.V;
else
    quadOrder = 7;
    lmax = models.utilities.matrixOperators.MoM2D.SMatrix.lmax(k0, mesh.nodes);
    [OPa, ~] = models.utilities.matrixOperators.MoM2D.batch.evaluate(mesh, f0, Zs, ...
        'normalize', false, 'symmetrize', true, 'quadOrder', quadOrder, ...
        'lmax', lmax, 'requests', {'R0', 'X0'});
    R0   = real(OPa.R0);
    X0   = real(OPa.X0);
    Lmat = full(models.utilities.matrixOperators.MoM2D.ohmicLosses.computeL(mesh, BF));

    % plane wave, normal incidence, unit E field
    planeWaveData.propagationVector = [0 0 1];
    planeWaveData.initElectricField = Epol;
    planeWaveData.axialRatio        = Inf;
    planeWaveData.direction         = 'right';
    V = models.solvers.MoM2D.excitation.planeWave(mesh, BF, planeWaveData, k0, quadOrder);

    save(cacheFile, 'mesh', 'BF', 'R0', 'X0', 'Lmat', 'V', 'f0', 'Zs', 'Epol');
    fprintf('Operators saved to %s\n', cacheFile);
end
R0   = (R0 + R0.')/2;
X0   = (X0 + X0.')/2;
Lmat = (Lmat + Lmat.')/2;

% Objective normalization: power incident on the design domain
Area = sum(mesh.triangleAreas);
Pinc = abs(E0)^2/(2*Z0vac)*Area;

Z0 = R0 + 1i*X0;
Z  = Z0 + Zs*Lmat;
Iful = Z \ (E0*V);
PsFull = 0.5*real(Iful'*R0*Iful)/Pinc;
fprintf('%s: N = %d, Area = %.4e m^2, Pinc = %.4e W, full structure Psca/Pinc = %.6e\n', ...
    structure, N, Area, Pinc, PsFull);

%% Dual bound + GCD (Python, scattered-voltage variable)
inFile  = [tempname '_dualIn.mat'];
outFile = [tempname '_dualOut.mat'];
save(inFile, 'R0', 'X0', 'Lmat', 'V', 'Zs', 'E0', 'Pinc', 'Ngcd', '-v7');
cmd = sprintf('"%s" "%s" "%s" "%s"', pythonExe, pyScript, inFile, outFile);
fprintf('Running %s\n', cmd);
status = system(cmd);
if status ~= 0
    error('dualGCDforMTB.py failed (status %d).', status);
end
dual = load(outFile);
delete(inFile, outFile);

Vopt = dual.Vopt(:);
LV   = (dual.LV + dual.LV')/2;
Iopt = dual.Iopt(:);

% Checks
PsOpt = 0.5*real(Iopt'*R0*Iopt)/Pinc;
fprintf('global bound %.6e, GCD bound %.6e, Psca(Iopt)/Pinc %.6e, full %.6e\n', ...
    dual.dualGlobal, dual.dualGCD, PsOpt, PsFull);
fprintf('min eig(LV) %.3e, ||-Z0*Iopt - Vopt||/||Vopt|| %.3e\n', ...
    min(eig(LV)), norm(-Z0*Iopt - Vopt)/norm(Vopt));

%% Operators for TSGA
% J = (Vsca - Vopt)' LV (Vsca - Vopt) with Vsca = -Z0*I on the whole design
% domain (I = 0 on removed edges) as a quadratic form in I:
% J = I' AJ I + 2 Re(bJ I) + cJ
OP.Mesh   = mesh;
OP.BF     = BF;
OP.f      = f0;
OP.fList  = f0;
OP.nFreqs = 1;
OP.Z      = Z;
OP.V      = E0*V;
OP.R0     = R0;
OP.Pinc   = Pinc;
AJ        = Z0'*LV*Z0;
OP.AJ     = (AJ + AJ')/2;
OP.bJ     = Vopt'*LV*Z0;
OP.cJ     = real(Vopt'*LV*Vopt);
OP.ports  = [];

[OP.M, OP.HbNorm, OP.HtNorm, OP.MnodeTria, OP.MnodeBasis] = models. ...
   utilities.matrixOperators.MoM2D.geometry.computeGraphMatrix(OP.Mesh, OP.BF);
OP.GeomBounds = models.utilities.matrixOperators.MoM2D. ...
   geometry.computeGeometryBounds(OP);

userData.reg = [0 0 0 0 0 0]; % no penalization of the shape
OP.userData  = userData;

fprintf('J(Iopt) = %.3e (should vanish), J(full) = %.6e\n', ...
    evalJ(OP, Iopt), evalJ(OP, Iful));

%% Optimization settings
optData.protectedEdges = [];
optData.fitness        = @ff_MO_scatVsDualDist;
optData.removingActive = true;
optData.addingActive   = true;
optData.edgesToCheck   = 0;    % 0 ~ ALL, 1 ~ all REM, BND ADD, 2 ~ BND
optData.nIters         = inf;  % max. number of iterations of TS
optData.relTol         = logspace(-2, -3.5, nIters);

% initial genes: empty, full, and thresholded dual optimal current
geneDual = abs(Iopt) > 0.1*max(abs(Iopt));
initGene = [zeros(1, N); ones(1, N); geneDual.'];

solverSettings.nAgents  = nAgents;
solverSettings.nIters   = nIters;
solverSettings.session  = 1;
solverSettings.relError = 1e-5;

AWsettings.Signes     = [-1, +1]; % max Psca/Pinc, min J
AWsettings.nObj       = length(AWsettings.Signes);
AWsettings.Weights    = TSmoo.uniformWeights(nAgents, AWsettings.nObj, 'MUD');
AWsettings.minIter    = 5;
AWsettings.iterPeriod = 3;
AWsettings.nadir      = [];
AWsettings.utop       = [];

solverSettings.AWopt = AWsettings;
solverSettings.AWalg = TSmoo.setAdaptiveWeights();

%% Pareto front (NSGA-II + TS)
% without a pool, parfor runs serially (the setting is restored afterwards)
ps = parallel.Settings;
autoCreate0 = ps.Pool.AutoCreate;
if ~useParallel
    ps.Pool.AutoCreate = false;
    delete(gcp('nocreate'));
end
try
    [ff, postData] = TS_BF.TSsolver.nsgaAndTopoOptGreedy(OP, optData, initGene, solverSettings);
catch err
    ps.Pool.AutoCreate = autoCreate0;
    rethrow(err);
end
ps.Pool.AutoCreate = autoCreate0;
[FallND, WallND, PallND] = TSmoo.extractNDfromPostData(postData);

PsND = -FallND(:, 1); % signed objectives -> Psca/Pinc
JND  = FallND(:, 2);
[PsND, iSort] = sort(PsND);
JND    = JND(iSort);
PallND = PallND(iSort, :);

% Psca/Pinc + J = dual bound holds for constraints satisfied by every design
% (global ones); the GCD projectors make the difference
fprintf('\n   Psca/Pinc            J    Psca/Pinc + J\n');
fprintf('%12.6e %12.4e %12.6e\n', [PsND, JND, PsND + JND].');

figure('Name', 'TSGA Pareto front');
plot(JND, PsND, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
hold on;
yline(dual.dualGCD, 'k-', 'GCD bound');
yline(dual.dualGlobal, 'k--', 'global bound');
plot(evalJ(OP, Iful), PsFull, 'rs', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('$J = (\mathbf{V}_\mathrm{sca} - \mathbf{V}_\mathrm{opt})^\mathrm{H} \mathbf{L}_V (\mathbf{V}_\mathrm{sca} - \mathbf{V}_\mathrm{opt})$', ...
    'Interpreter', 'latex');
ylabel('$P_\mathrm{sca}/P_\mathrm{inc}$', 'Interpreter', 'latex');
legend('TSGA Pareto front', 'GCD bound', 'global bound', 'full structure', 'Location', 'best');
grid on;

%% Extreme designs of the front
optimEdges = setdiff(1:N, optData.protectedEdges);
for k = [numel(PsND), 1] % max Psca/Pinc and min J
    MO_iter  = PallND(k, 1);
    MO_agent = PallND(k, 2);
    gene       = postData.GenusFinal(MO_agent, :, MO_iter);
    globIndsBF = sort([optimEdges(find(gene)), optData.protectedEdges]);
    I = TSaux.assemblyCurrentI(OP, globIndsBF);
    fprintf('design %d: Psca/Pinc = %.6e, J = %.4e, %d of %d edges\n', k, ...
        0.5*real(I'*R0*I)/Pinc, evalJ(OP, I), numel(globIndsBF), N);
    TSaux.plotStructure(OP.Mesh, OP.BF, optData.protectedEdges, globIndsBF, false);
    title(sprintf('Psca/Pinc = %.4e, J = %.3e', PsND(k), JND(k)));
    results.plotCurrent(OP.Mesh, 'basisFcns', OP.BF, 'Ivec', I, ...
        'part', 'abs', 'arrowScale', 'proportional');
    title(sprintf('|I|, Psca/Pinc = %.4e', PsND(k)));
end

% Dual optimal current
results.plotCurrent(OP.Mesh, 'basisFcns', OP.BF, 'Ivec', Iopt, ...
    'part', 'abs', 'arrowScale', 'proportional');
title(sprintf('|I_{opt}|, GCD bound %.4e', dual.dualGCD));

save(fullfile(thisDir, ['TSGAdualPareto_' tag '.mat']), ...
    'dual', 'PsFull', 'Pinc', 'postData', 'FallND', 'WallND', 'PallND', ...
    'PsND', 'JND', 'solverSettings', 'optData');

%% Local functions
function J = evalJ(OP, I)
% distance of the scattered voltage of current I from the dual optimum
J = real(I'*OP.AJ*I) + 2*real(OP.bJ*I) + OP.cJ;
end

function [FF, paramValue] = ff_MO_scatVsDualDist(...
   OP, I, globIndsBF, globIndsBF_tested, BFtype, signedWeights, giter)
% Two objectives: [Psca/Pinc, J], J distance of Vsca from the dual optimum

Psca = 0.5*real(TS_BF.TSeval.evaluateLinQuadForm(...
   OP.R0, [], I, globIndsBF, globIndsBF_tested, BFtype))/OP.Pinc;
IAI  = real(TS_BF.TSeval.evaluateLinQuadForm(...
   OP.AJ, [], I, globIndsBF, globIndsBF_tested, BFtype));
if BFtype == -1 || BFtype == 0
    bI = TS_BF.TSeval.evaluateLinearForm(OP.bJ, I{1}, globIndsBF);
else
    bI = TS_BF.TSeval.evaluateLinearForm(OP.bJ, I{1}, globIndsBF, globIndsBF_tested);
end
J = IAI + 2*real(bI(:)) + OP.cJ;

MOF = [Psca, J];
FF  = sum(signedWeights .* MOF, 2);

if nargout > 1
   paramValue.MOF  = MOF; % (mandatory)
   paramValue.Psca = Psca;
   paramValue.J    = J;
end
end
