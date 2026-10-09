%% Minimum Q of a PEC 2:1 plate fed by a 50 Ohm port, dual bound with GCD
% Minimizes the stored energy (omW) of a PEC structure inside a 2:1
% rectangle, ka = 1, fed by a delta gap at the basis function nearest the
% midpoint of the longer edge. The input voltage 1 V and the input
% impedance 50 Ohm fix the input current I0 = 20 mA and the radiated power
% P = 10 mW, so Q = 25 I'*omW*I for a self-resonant current.
%
% The dual bound with GCD refinement is evaluated by minQportGCD.py in the
% scattered-voltage variable Vsca = -Z0*I (run on Narval by
% narval_minQportGCD.sh). The structure is inferred from the dual optimal
% current as z_n = (V + Vsca)_n/I_n, metal |z| -> 0, void |z| -> Inf.
%
% stage = 'assemble' : mesh, normalized operators, port -> *_in.mat
% stage = 'post'     : *_out.mat -> inferred structure, Q, Zin, figures
% source = 'local'   : infer from the all-local bound current (IoptLocal)
% source = 'gcd'     : infer from the GCD dual optimal current (Iopt)
%
% Requires AToM (with the TopologySensitivity package) on the path.

if ~exist('stage', 'var')
    stage = 'assemble';
end
if ~exist('source', 'var')
    source = 'local';
end
close all;

%% Settings
if ~exist('m', 'var')
    m = 25;                 % EuCAP27 ortho mesh, 25 -> N = 285, 50 -> N = 744
end
tag         = 'minQport_patch21ka1';
if m ~= 25
    tag = sprintf('%s_m%d', tag, m);
end
ka          = 1;
fixMeshToKa = 2;
f0          = 2.45e9;       % frequency (Hz)
Vp          = 1;            % input voltage (V)
Zin0        = 50;           % required input impedance (Ohm)
I0          = Vp/Zin0;      % input current (A)
if ~exist('Ngcd', 'var')
    Ngcd = 50;              % number of GCD iterations
end

thisDir = fileparts(mfilename('fullpath'));
inFile  = fullfile(thisDir, [tag '_in.mat']);
if ~exist('runTag', 'var')
    runTag = '';            % e.g. '_gcd20' -> *_gcd20_out.mat, *_gcd20_<source>_* figures
end
outFile = fullfile(thisDir, [tag runTag '_out.mat']);

%% Mesh (EuCAP27/code/getMatricesFixedMesh.m)
c0  = models.utilities.constants.c0;
k0  = 2*pi*f0/c0;
lam = c0/f0;
a    = ka/k0;
len  = 4/sqrt(5)*a;
wid  = len/2;
mEff = m*fixMeshToKa/ka;
Ny   = ceil((wid/lam*mEff + 1)/2);
[~, ~, meshPx] = models.utilities.meshPublic.pixelGridToOrthoMesh(2*ones(Ny, 2*Ny), 1);
nodes = meshPx.nodes./meshPx.normDistanceA*sqrt((wid/2)^2 + (len/2)^2);
mesh  = models.utilities.meshPublic.getMeshData2D(nodes, meshPx.connectivityList);
h     = len/(2*Ny);         % pixel size

switch stage
    case 'assemble'
        %% Operators (normalized -> Ohm, port voltage in V, terminal current in A)
        quadOrder = 7;
        lmax = models.utilities.matrixOperators.MoM2D.SMatrix.lmax(k0, mesh.nodes);
        [OP, ~] = models.utilities.matrixOperators.MoM2D.batch.evaluate(mesh, f0, 0, ...
            'normalize', true, 'symmetrize', true, 'quadOrder', quadOrder, ...
            'lmax', lmax, 'requests', {'R0', 'X0', 'omW'});
        BF  = OP.BF;
        N   = BF.nUnknowns;
        R0  = real(OP.R0);  R0  = (R0 + R0.')/2;
        X0  = real(OP.X0);  X0  = (X0 + X0.')/2;
        omW = real(OP.omW); omW = (omW + omW.')/2;

        %% Port: BF nearest the midpoint of the longer (bottom, y = -wid/2) edge
        feederPosition = [0, min(mesh.nodes(:, 2)) + h/2, 0];
        edgeCenters = mesh.triangleEdgeCenters(BF.data(:, 3), :);
        [dist, port] = min(models.utilities.vectorOperations.rowNorm(edgeCenters - feederPosition));
        fprintf('N = %d, size %.4f x %.4f m, port BF %d at [%.4f %.4f] (%.1e m from target)\n', ...
            N, len, wid, port, edgeCenters(port, 1:2), dist);
        fprintf('min eig: omW %.3e, R0 %.3e\n', min(eig(omW)), min(eig(R0)));

        % full plate fed at the port (reference)
        V = zeros(N, 1); V(port) = Vp;
        Z0 = R0 + 1i*X0;
        Ifull = Z0\V;
        [Qfull, ZinFull] = portQ(Ifull, port, Vp, R0, X0, omW);
        fprintf('full plate: Zin = %.3f %+.3fj Ohm, Q = %.4g\n', real(ZinFull), imag(ZinFull), Qfull);

        hndl = results.plotMesh(mesh, 'options', struct('showEdges', true, 'showTriangles', true));
        set(hndl.edges(BF.data(port, 3)), 'Color', [1 0 0], 'LineWidth', 2);
        title(sprintf('N = %d, port BF %d', N, port));
        exportgraphics(gcf, fullfile(thisDir, [tag '_port.png']));

        port = port - 1; %#ok<NASGU> zero based in Python
        save(inFile, 'R0', 'X0', 'omW', 'port', 'I0', 'Vp', 'Ngcd');
        port = port + 1;
        save(fullfile(thisDir, [tag '_operators.mat']), 'mesh', 'BF', 'R0', 'X0', 'omW', ...
            'port', 'Qfull', 'ZinFull');
        fprintf('saved %s\n', inFile);

    case 'post'
        %% Dual optimum
        ops = load(fullfile(thisDir, [tag '_operators.mat']));
        if isfile(outFile)
            out = load(outFile);
        else % GCD job not finished yet
            out = struct('QlbGlobal', NaN, 'QlbGCD', NaN, 'QlbLocal', NaN, 'Iopt', NaN);
        end
        localFile = fullfile(thisDir, [tag '_localout.mat']);
        if isfile(localFile) % all-local bound from a separate LOCAL_ONLY job
            loc = load(localFile);
            out.QlbLocal  = loc.QlbLocal;
            out.IoptLocal = loc.IoptLocal;
        end
        BF = ops.BF; R0 = ops.R0; X0 = ops.X0; omW = ops.omW; port = ops.port;
        N  = BF.nUnknowns;
        Z0 = R0 + 1i*X0;
        V  = zeros(N, 1); V(port) = Vp;
        switch source
            case 'local'
                Iopt = out.IoptLocal(:);
                Qlb  = out.QlbLocal;
            case 'gcd'
                Iopt = out.Iopt(:);
                Qlb  = out.QlbGCD;
        end
        postTag = [tag runTag '_' source];
        Vsca = -Z0*Iopt;
        [QIopt, ZinIopt] = portQ(Iopt, port, Vp, R0, X0, omW);
        fprintf('bounds: global %.4g, GCD %.4g, local %.4g, full plate %.4g\n', ...
            out.QlbGlobal, out.QlbGCD, out.QlbLocal, ops.Qfull);
        fprintf('Iopt (%s): Zin = %.3f %+.3fj Ohm, Q = %.4g\n', source, ...
            real(ZinIopt), imag(ZinIopt), QIopt);

        %% Inference, z = (V + Vsca)./I (metal -> 0, void -> Inf)
        zInf = (V + Vsca)./Iopt;
        zInf(port) = 0;
        [~, order] = sort(abs(zInf));
        nList = 2:N;
        QList = nan(size(nList)); ZinList = nan(size(nList)); GammaList = nan(size(nList));
        for iN = 1:numel(nList)
            S = sort(order(1:nList(iN)));
            I = zeros(N, 1);
            I(S) = Z0(S, S)\V(S);
            [QList(iN), ZinList(iN)] = portQ(I, port, Vp, R0, X0, omW);
            GammaList(iN) = abs((ZinList(iN) - Zin0)/(ZinList(iN) + Zin0));
        end
        matched = GammaList < 1/3;
        if any(matched)
            Qm = QList; Qm(~matched) = Inf;
            [~, iBest] = min(Qm);
        else
            [~, iBest] = min(QList);
        end
        [~, iBestQ] = min(QList);
        S = sort(order(1:nList(iBest)));
        Ibest = zeros(N, 1); Ibest(S) = Z0(S, S)\V(S);
        fprintf('inferred (|Gamma|<1/3): %d BFs, Q = %.4g, Zin = %.3f %+.3fj, |Gamma| = %.3f\n', ...
            nList(iBest), QList(iBest), real(ZinList(iBest)), imag(ZinList(iBest)), GammaList(iBest));
        fprintf('inferred (min Q):       %d BFs, Q = %.4g, Zin = %.3f %+.3fj, |Gamma| = %.3f\n', ...
            nList(iBestQ), QList(iBestQ), real(ZinList(iBestQ)), imag(ZinList(iBestQ)), GammaList(iBestQ));

        save(fullfile(thisDir, [postTag '_post.mat']), 'zInf', 'order', 'nList', 'QList', ...
            'ZinList', 'GammaList', 'iBest', 'iBestQ', 'S', 'Ibest', 'QIopt', 'ZinIopt');

        %% Figures
        bfX = mesh.triangleEdgeCenters(BF.data(:, 3), 1);
        bfY = mesh.triangleEdgeCenters(BF.data(:, 3), 2);

        figure('Position', [100 100 900 600]);
        subplot(2, 1, 1);
        scatter(bfX, bfY, 40, log10(abs(zInf) + eps), 'filled'); axis equal; colorbar;
        hold on; plot(bfX(port), bfY(port), 'rx', 'MarkerSize', 12, 'LineWidth', 2);
        title('log_{10}|z_n|, z = (V + V_{sca})/I_{opt}');
        subplot(2, 1, 2);
        scatter(bfX, bfY, 40, abs(Iopt), 'filled'); axis equal; colorbar;
        title('|I_{opt}| (A)');
        exportgraphics(gcf, fullfile(thisDir, [postTag '_inference.png']));

        figure;
        hQ = semilogy(nList, QList, 'o-'); hold on;
        hM = semilogy(nList(matched), QList(matched), 'g.', 'MarkerSize', 14);
        if isfinite(out.QlbGCD), yline(out.QlbGCD, 'r--', 'GCD bound'); end
        if isfinite(out.QlbGlobal), yline(out.QlbGlobal, 'k:', 'global bound'); end
        if isfinite(out.QlbLocal), yline(out.QlbLocal, 'b-.', 'local bound'); end
        xlabel('number of metal BFs (sorted by |z|)'); ylabel('Q');
        if any(matched)
            legend([hQ hM], {'inferred', '|\Gamma| < 1/3'}, 'Location', 'best');
        else
            legend(hQ, 'inferred (none with |\Gamma| < 1/3)', 'Location', 'best');
        end
        grid on;
        exportgraphics(gcf, fullfile(thisDir, [postTag '_Qsweep.png']));

        % inferred structure (BFs kept as metal, removed BFs are slots)
        TSaux.plotStructure(mesh, BF, port, S, false);
        title(sprintf('inferred (%s): %d/%d BFs, Q = %.3g (bound %.3g), Z_{in} = %.1f %+.1fj \\Omega', ...
            source, numel(S), N, QList(iBest), Qlb, real(ZinList(iBest)), imag(ZinList(iBest))));
        exportgraphics(gcf, fullfile(thisDir, [postTag '_structure.png']));

        results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', Ibest, ...
            'part', 'abs', 'arrowScale', 'proportional');
        title(sprintf('|J| of the inferred structure, Q = %.3g', QList(iBest)));
        exportgraphics(gcf, fullfile(thisDir, [postTag '_structureCurrent.png']));

        results.plotCurrent(mesh, 'basisFcns', BF, 'iVec', Iopt, ...
            'part', 'abs', 'arrowScale', 'proportional');
        title(sprintf('|J_{opt}| (%s), Q = %.3g, bound %.3g', source, QIopt, Qlb));
        exportgraphics(gcf, fullfile(thisDir, [postTag '_dualCurrent.png']));
end

function [Q, Zin] = portQ(I, port, Vp, R0, X0, omW)
% Q = (I'omW I + |I'X0 I|)/(2 I'R0 I) (minQselfRes), Zin = Vp/I_port
Q   = 0.5*real(I'*omW*I + abs(I'*X0*I))/real(I'*R0*I);
Zin = Vp/I(port);
end
