%% Stored-energy operator omW of the 2:1 patch, ka = 1, f = 2.45 GHz (*_patch21ka1.txt)
% Rebuilds the mesh of Lmat/R0/X0/V_patch21ka1.txt (hexagonal pixel grid
% 7 x 12, equilateral triangles, N = 233 unnormalized RWG functions, scaled
% to ka = 1, y-polarized plane wave) and exports the stored-energy matrix
%   omW = omega*(dX0/domega) - radiation term (Vandenbosch),
% normalized as Q = I'*omW*I/(2 I'*R0*I) for a self-resonant current, i.e.
% omega (Wm + We) = I'*omW*I/4 (AToM 'omW', as minQportGCD.m).
% The stored operators are checked against the regenerated ones (quadOrder 12
% reproduces X0_patch21ka1.txt to round-off, quadOrder 7 does not).
%
% Requires AToM on the path. Writes omW_patch21ka1.txt and the mesh
% (mesh_patch21ka1.mat: nodes [m], triangles and RWG table, 0 based, for
% plotting structures in Python) next to this script.

thisDir   = fileparts(mfilename('fullpath'));
matrixSet = 'patch21ka1';
f0        = 2.45e9;
quadOrder = 12;

copper_conductivity = 5.96e7;
omega = 2*pi*f0;
delta = sqrt(2/(omega*copper_conductivity*models.utilities.constants.mu0));
Zs    = (1 + 1i)/(copper_conductivity*delta);

k0 = 2*pi*f0/models.utilities.constants.c0;
a  = 1/k0;                                   % ka = 1

[~, ~, mesh] = models.utilities.meshPublic.pixelGridToHexaMesh(ones(7, 12), 1);
mesh.nodes = mesh.nodes./mesh.normDistanceA*a;
mesh = models.utilities.meshPublic.getMeshData2D(mesh.nodes, mesh.connectivityList);

lmax = models.utilities.matrixOperators.MoM2D.SMatrix.lmax(k0, mesh.nodes);
[OP, ~] = models.utilities.matrixOperators.MoM2D.batch.evaluate(mesh, f0, Zs, ...
    'normalize', false, 'symmetrize', true, 'quadOrder', quadOrder, 'lmax', lmax, ...
    'requests', {'R0', 'X0', 'Lmat', 'omW'});

planeWaveData.propagationVector = [0 0 1];
planeWaveData.initElectricField = [0 1 0];
planeWaveData.axialRatio = Inf;
planeWaveData.direction = 'right';
V = models.solvers.MoM2D.excitation.planeWave(mesh, OP.BF, planeWaveData, k0, quadOrder);

%% Consistency with the stored operators
relErr = @(A, B) norm(A - B, 'fro')/norm(B, 'fro');
stored = struct('Lmat', full(OP.Lmat), 'R0', OP.R0, 'X0', OP.X0, 'V', V);
for name = fieldnames(stored).'
    ref = readmatrix(fullfile(thisDir, sprintf('%s_%s.txt', name{1}, matrixSet)));
    fprintf('%-4s: ||new - stored|| / ||stored|| = %.2e\n', name{1}, relErr(stored.(name{1}), ref));
end

omW = real(OP.omW);
omW = (omW + omW.')/2;
ev  = eig(omW);
fprintf('N = %d, eig(omW) in [%.3e, %.3e]\n', OP.BF.nUnknowns, min(ev), max(ev));

writematrix(omW, fullfile(thisDir, sprintf('omW_%s.txt', matrixSet)), 'Delimiter', ',');
fprintf('saved omW_%s.txt\n', matrixSet);

%% Mesh for plotting (0 based): RWG n spans trianglePlus/triangleMinus across edge (nodes edgeNodes)
BFdata    = OP.BF.data;
nodes     = mesh.nodes(:, 1:2);
triangles = mesh.connectivityList - 1;
bf        = [BFdata(:, OP.BF.Col.trianglePlus), BFdata(:, OP.BF.Col.triangleMinus), ...
             mesh.edges(BFdata(:, OP.BF.Col.edge), :)] - 1; % [T+, T-, edge node 1, edge node 2]
save(fullfile(thisDir, sprintf('mesh_%s.mat', matrixSet)), 'nodes', 'triangles', 'bf');
fprintf('saved mesh_%s.mat\n', matrixSet);
