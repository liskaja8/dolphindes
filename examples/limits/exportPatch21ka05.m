%% Operators of the 2:1 patch at ka = 0.5, f = 2.45 GHz (*_patch21ka05.txt)
% Same hexagonal mesh as exportOmWpatch21ka1.m (7 x 12, N = 233), scaled to ka = 0.5;
% y-polarized plane wave (along the short edge), quadOrder 12 as for patch21ka1.
% Writes Lmat, R0, X0, V, omW _patch21ka05.txt and mesh_patch21ka05.mat (nodes [m],
% triangles and RWG table, 0 based) next to this script. Requires AToM on the path.

thisDir   = fileparts(mfilename('fullpath'));
matrixSet = 'patch21ka05';
f0        = 2.45e9;
quadOrder = 12;

copper_conductivity = 5.96e7;
omega = 2*pi*f0;
delta = sqrt(2/(omega*copper_conductivity*models.utilities.constants.mu0));
Zs    = (1 + 1i)/(copper_conductivity*delta);

k0 = 2*pi*f0/models.utilities.constants.c0;
a  = 0.5/k0;                                 % ka = 0.5

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

%% Export of the operators
for name = {'Lmat', 'R0', 'X0'}
    writematrix(full(OP.(name{1})), fullfile(thisDir, sprintf('%s_%s.txt', name{1}, matrixSet)), 'Delimiter', ',');
end
writematrix(V, fullfile(thisDir, sprintf('V_%s.txt', matrixSet)), 'Delimiter', ',');
fprintf('saved Lmat, R0, X0, V _%s.txt\n', matrixSet);

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
