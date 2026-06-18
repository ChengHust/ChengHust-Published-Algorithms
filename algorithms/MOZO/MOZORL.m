function result = MOZORL(problem, options)
%
%   result = MOZORL(problem, options)
%
%   Required problem fields:
%       objective   Function handle, f = objective(x), minimized objectives.
%       lower       1-by-D lower bound.
%       upper       1-by-D upper bound.
%
%   Optional problem fields:
%       nvars       Number of decision variables. Inferred from lower.
%       nobjs       Number of objectives. Inferred from the first evaluation.
%
%   Common options fields:
%       populationSize   Population size, default 10.
%       archiveSize      Archive size, default populationSize.
%       maxEvaluations   Evaluation budget, default 1000.
%       maxGenerations   Generation budget, default Inf.
%       estimation       1 random forward, 2 random central, 3/4 sign forms.
%       aggregation      1 DAG, 2 CONFES, 3 ConFIG for two objectives.
%       acceleration     1 conjugate-gradient, 2 Adam, 3 none.
%       searchRadius     Integer line-search radius, default 5.
%       mu               Zeroth-order perturbation size, default 1e-3.
%       step             Line-search base, default 0.5.
%       agent            Optional RL agent or action function.
%       verbose          Print progress, default true.
%
%   Example:
%       problem.objective = @(x) [sum(x.^2), sum((x-1).^2)];
%       problem.lower = -5 * ones(1, 20);
%       problem.upper =  5 * ones(1, 20);
%       result = MOZORL(problem, struct("maxEvaluations", 2000));

    if nargin < 2 || isempty(options)
        options = struct();
    end

    validateProblem(problem);
    options = setDefaults(options, problem);
    if isfield(options, "seed") && ~isempty(options.seed)
        rng(options.seed);
    end

    lower = rowvec(problem.lower);
    upper = rowvec(problem.upper);
    D = numel(lower);
    NP = options.populationSize;
    archiveSize = options.archiveSize;

    W = uniformWeights(NP, options.nobjs);
    [population, evalCount] = initializePopulation(problem, lower, upper, NP);
    archive = updateArchive(population, archiveSize);
    refPoint = calculateReferencePoint(archive);
    state = calculateMOZOState(population, archive, D, evalCount, options.maxEvaluations, refPoint);

    K = zeros(1, NP);
    g0 = cell(1, NP);
    d0 = cell(1, NP);
    m0 = zeros(NP, D);
    v0 = zeros(NP, D);
    accStorage = options.acceleration;
    generation = 0;
    history = struct("generation", {}, "evaluations", {}, "bestObjective", {}, "archiveSize", {});

    while evalCount < options.maxEvaluations && generation < options.maxGenerations
        generation = generation + 1;

        if ~isempty(options.agent)
            action = agentAction(options.agent, state);
            params = decodeAction(action);
        else
            params = defaultMOZOParameters(options);
        end

        if params.acceleration ~= accStorage
            [K, g0, d0, m0, v0] = resetAccelerationState(params.acceleration, NP, D);
            accStorage = params.acceleration;
        end

        [population, archive, K, g0, d0, m0, v0, evalCount] = executeMOZOIteration( ...
            problem, lower, upper, population, archive, W, archiveSize, params, ...
            K, g0, d0, m0, v0, evalCount, options.maxEvaluations);

        state = calculateMOZOState(population, archive, D, evalCount, options.maxEvaluations, refPoint);
        bestObj = min(cat(1, archive.obj), [], 1);
        history(end + 1) = struct( ...
            "generation", generation, ...
            "evaluations", evalCount, ...
            "bestObjective", bestObj, ...
            "archiveSize", numel(archive)); %#ok<AGROW>

        if options.verbose
            fprintf("Generation %d | FE %d | best objectives: %s\n", ...
                generation, evalCount, mat2str(bestObj, 5));
        end
    end

    [~, bestIdx] = min(arrayfun(@(p) p.obj(1), archive));
    result = struct();
    result.bestDecision = archive(bestIdx).dec;
    result.bestObjective = archive(bestIdx).obj;
    result.archive = archive;
    result.history = history;
    result.evaluations = evalCount;
    result.generations = generation;
    result.referencePoint = refPoint;
end

function validateProblem(problem)
    required = ["objective", "lower", "upper"];
    for i = 1:numel(required)
        if ~isfield(problem, required(i))
            error("MOZORL:InvalidProblem", "Missing problem.%s.", required(i));
        end
    end
    if ~isa(problem.objective, "function_handle")
        error("MOZORL:InvalidProblem", "problem.objective must be a function handle.");
    end
    if numel(problem.lower) ~= numel(problem.upper)
        error("MOZORL:InvalidProblem", "problem.lower and problem.upper must have the same length.");
    end
end

function options = setDefaults(options, problem)
    lower = rowvec(problem.lower);
    probe = rowvec(problem.objective((lower + rowvec(problem.upper)) / 2));

    defaults = struct();
    defaults.populationSize = 10;
    defaults.archiveSize = [];
    defaults.maxEvaluations = 1000;
    defaults.maxGenerations = inf;
    defaults.estimation = 2;
    defaults.aggregation = 2;
    defaults.acceleration = 1;
    defaults.searchRadius = 5;
    defaults.mu = 1e-3;
    defaults.step = 0.5;
    defaults.agent = [];
    defaults.verbose = true;
    defaults.nobjs = numel(probe);

    names = fieldnames(defaults);
    for i = 1:numel(names)
        name = names{i};
        if ~isfield(options, name) || isempty(options.(name))
            options.(name) = defaults.(name);
        end
    end
    if isfield(problem, "nobjs") && ~isempty(problem.nobjs)
        options.nobjs = problem.nobjs;
    end
    if isempty(options.archiveSize)
        options.archiveSize = options.populationSize;
    end
end

function params = defaultMOZOParameters(options)
    params = struct();
    params.estimation = options.estimation;
    params.aggregation = options.aggregation;
    params.acceleration = options.acceleration;
    params.searchRadius = options.searchRadius;
    params.mu = options.mu;
    params.step = options.step;
end

function [population, evalCount] = initializePopulation(problem, lower, upper, NP)
    D = numel(lower);
    population = repmat(struct("dec", [], "obj", []), 1, NP);
    evalCount = 0;
    if isfield(problem, "initialPopulation") && ~isempty(problem.initialPopulation)
        initial = problem.initialPopulation;
        if size(initial, 1) < NP
            error("MOZORL:InvalidProblem", "problem.initialPopulation must contain at least populationSize rows.");
        end
        decisions = initial(1:NP, :);
    else
        decisions = lower + rand(NP, D) .* (upper - lower);
    end
    for i = 1:NP
        population(i) = evaluateDecision(problem, decisions(i, :), lower, upper);
        evalCount = evalCount + 1;
    end
end

function solution = evaluateDecision(problem, x, lower, upper)
    x = min(max(rowvec(x), lower), upper);
    solution = struct("dec", x, "obj", rowvec(problem.objective(x)));
end

function [population, archive, K, g0, d0, m0, v0, evalCount] = executeMOZOIteration( ...
    problem, lower, upper, population, archive, W, archiveSize, params, ...
    K, g0, d0, m0, v0, evalCount, maxEvaluations)

    NP = numel(population);
    D = numel(population(1).dec);
    alpha = 1;
    beta1 = 0.9;
    beta2 = 0.999;
    offspring = repmat(struct("dec", [], "obj", []), 1, 0);

    if params.acceleration == 1
        K = mod(K, D) + 1;
    end

    for i = 1:NP
        if evalCount >= maxEvaluations
            break;
        end

        [gk, site1, evalCount] = finiteDifference(problem, lower, upper, ...
            population(i), W(i, :), params, evalCount, maxEvaluations);

        if params.acceleration == 1
            if K(i) == 1 || isempty(g0{i}) || isempty(d0{i})
                dk = -gk;
            else
                beta = (gk' * gk) / (g0{i}' * g0{i} + 1e-12);
                dk = -gk + beta * d0{i};
                if gk' * dk >= 0
                    dk = -gk;
                end
            end
        elseif params.acceleration == 2
            K(i) = K(i) + 1;
            m0(i, :) = beta1 * m0(i, :) + (1 - beta1) * gk';
            v0(i, :) = beta2 * v0(i, :) + (1 - beta2) * (gk'.^2);
            dk = (-alpha * (m0(i, :) / (1 - beta1^K(i))) ./ ...
                (sqrt(v0(i, :) / (1 - beta2^K(i))) + 1e-8))';
        else
            dk = -gk;
        end

        site2 = abs(dk) <= 1e-12;
        success = false;
        candidate = population(i);
        for exponent = -params.searchRadius:params.searchRadius
            if evalCount >= maxEvaluations
                break;
            end
            offDec = makeOffspring(population(i), archive, dk, site1, site2, params, exponent);
            candidate = evaluateDecision(problem, offDec, lower, upper);
            evalCount = evalCount + 1;
            if any(candidate.obj < population(i).obj)
                offspring(end + 1) = candidate; %#ok<AGROW>
            end
            if all(candidate.obj < population(i).obj)
                success = true;
                break;
            end
        end

        if success
            population(i) = candidate;
            if params.acceleration == 1
                g0{i} = gk;
                d0{i} = dk;
            end
        else
            population(i) = archive(randi(numel(archive)));
            if params.acceleration == 1
                K(i) = 0;
                g0{i} = [];
                d0{i} = [];
            elseif params.acceleration == 2
                m0(i, :) = 0;
                v0(i, :) = 0;
            end
        end
    end

    archive = updateArchive([archive, offspring], archiveSize);
end

function offDec = makeOffspring(parent, archive, dk, site1, site2, params, exponent)
    D = numel(parent.dec);
    archiveA = archive(randi(numel(archive))).dec;
    archiveB = archive(randi(numel(archive))).dec;
    stepSize = params.step^exponent;
    mutationProb = 0;
    if any(site1)
        mutationProb = min(1, 1 / nnz(site1));
    end
    mutation = rand(1, D) < mutationProb;
    direction = dk';

    if params.aggregation == 1
        offDec = parent.dec + (~site2') .* stepSize .* direction + ...
            mutation .* site2' .* stepSize .* (archiveA - archiveB);
    elseif params.aggregation == 2
        offDec = parent.dec + (~site2' & ~site1) .* stepSize .* direction + ...
            mutation .* (site2' | site1) .* stepSize .* (archiveA - archiveB);
    else
        offDec = parent.dec + (~site2') .* stepSize .* direction + ...
            mutation .* site2' .* stepSize .* (archiveA - archiveB);
    end
end

function [df, site, evalCount] = finiteDifference(problem, lower, upper, X, W, params, evalCount, maxEvaluations)
    D = numel(X.dec);
    direction = randn(1, D);
    direction = direction / (norm(direction) + 1e-12);

    if any(params.estimation == [2, 4, 6]) && evalCount + 2 <= maxEvaluations
        plus = evaluateDecision(problem, X.dec + params.mu * direction, lower, upper);
        minus = evaluateDecision(problem, X.dec - params.mu * direction, lower, upper);
        evalCount = evalCount + 2;
        gradObj = (plus.obj - minus.obj) / (2 * params.mu);
    else
        plus = evaluateDecision(problem, X.dec + params.mu * direction, lower, upper);
        evalCount = evalCount + 1;
        gradObj = (plus.obj - X.obj) / params.mu;
    end

    gradients = D * direction' * gradObj;
    if any(params.estimation == [3, 4])
        gradients = sign(gradients);
    end

    site = (any(gradients < 0, 2) & any(gradients > 0, 2))';
    if params.aggregation == 3 && size(gradients, 2) == 2
        df = configDouble(gradients(:, 1), gradients(:, 2));
    else
        df = gradients * rowvec(W)';
    end
end

function state = calculateMOZOState(population, archive, D, evalCount, maxEvaluations, refPoint)
    popObjs = cat(1, population.obj);
    archiveObjs = cat(1, archive.obj);

    hv = calculateHypervolume(archiveObjs, refPoint);
    maxHV = prod(max(refPoint - min(archiveObjs, [], 1), 0));
    state1 = safeRatio(hv, maxHV);

    state2 = 0;
    if size(archiveObjs, 1) > 1
        distances = pairwiseDistances(archiveObjs, archiveObjs, true);
        state2 = safeRatio(std(distances), mean(distances));
    end

    state3 = 1;
    if size(archiveObjs, 1) > 1 && size(popObjs, 1) > 1
        popArchiveDist = pairwiseDistances(popObjs, archiveObjs, false);
        archiveDist = pairwiseDistances(archiveObjs, archiveObjs, true);
        state3 = safeRatio(mean(min(popArchiveDist, [], 2)), mean(archiveDist));
    end

    state4 = numel(archive) / max(numel(population), 1);
    state5 = 0;
    if size(popObjs, 1) > 1
        popDist = pairwiseDistances(popObjs, popObjs, true);
        state5 = safeRatio(mean(popDist), max(popDist));
    end

    maxPop = max(popObjs, [], 1);
    minArchive = min(archiveObjs, [], 1);
    state6 = mean(max(0, maxPop - minArchive) ./ (abs(maxPop) + 1e-10));
    state7 = max(0, 1 - evalCount / maxEvaluations);
    state8 = min(log10(D) / log10(100000), 1);
    state = [state1; state2; state3; state4; state5; state6; state7; state8];
end

function action = agentAction(agent, state)
    if isa(agent, "function_handle")
        action = agent(state);
    elseif exist("getAction", "file") == 2
        action = getAction(agent, state);
        if iscell(action)
            action = action{1};
        end
    else
        error("MOZORL:AgentUnavailable", ...
            "Pass options.agent as a function handle, or provide a Reinforcement Learning Toolbox agent.");
    end
end

function params = decodeAction(action)
    action = min(max(rowvec(action), -1), 1);
    params = struct();
    params.estimation = min(max(round((action(1) + 1) * 2.5) + 1, 1), 6);
    params.aggregation = min(max(round((action(2) + 1) * 1) + 1, 1), 3);
    params.acceleration = min(max(round((action(3) + 1) * 1) + 1, 1), 3);
    params.searchRadius = min(max(round((action(4) + 1) * 4.5) + 1, 1), 10);
    params.mu = 10^(action(5) * 2 - 4);
    params.step = 10^(action(6) * 2);
end

function [K, g0, d0, m0, v0] = resetAccelerationState(acceleration, NP, D)
    K = zeros(1, NP);
    g0 = cell(1, NP);
    d0 = cell(1, NP);
    m0 = zeros(NP, D);
    v0 = zeros(NP, D);
    if acceleration ~= 1
        g0(:) = {[]};
        d0(:) = {[]};
    end
end

function archive = updateArchive(population, N)
    objs = cat(1, population.obj);
    front = firstFront(objs);
    archive = population(front);
    if numel(archive) > N
        archiveObjs = cat(1, archive.obj);
        choose = true(1, numel(archive));
        distances = pairwiseDistances(archiveObjs, archiveObjs, false);
        distances(eye(size(distances)) == 1) = inf;
        while nnz(choose) > N
            remain = find(choose);
            sortedDistances = sort(distances(remain, remain), 2);
            [~, rank] = sortrows(sortedDistances);
            choose(remain(rank(1))) = false;
        end
        archive = archive(choose);
    end
end

function front = firstFront(objs)
    n = size(objs, 1);
    dominated = false(n, 1);
    for i = 1:n
        for j = 1:n
            if i ~= j && all(objs(j, :) <= objs(i, :)) && any(objs(j, :) < objs(i, :))
                dominated(i) = true;
                break;
            end
        end
    end
    front = find(~dominated)';
end

function refPoint = calculateReferencePoint(archive)
    objs = cat(1, archive.obj);
    maxObjs = max(objs, [], 1);
    minObjs = min(objs, [], 1);
    margin = 0.1 * (maxObjs - minObjs + 1);
    refPoint = maxObjs + margin;
end

function hv = calculateHypervolume(points, refPoint)
    if isempty(points)
        hv = 0;
        return;
    end
    [N, M] = size(points);
    fmin = min(min(points, [], 1), zeros(1, M));
    fmax = max(refPoint, [], 1);
    denom = (fmax - fmin) * 1.1;
    denom(denom == 0) = 1;
    popObj = (points - repmat(fmin, N, 1)) ./ repmat(denom, N, 1);
    popObj(any(popObj > 1, 2), :) = [];
    ref = ones(1, M);
    if isempty(popObj)
        hv = 0;
    elseif M < 4
        pl = sortrows(popObj);
        S = {1, pl};
        for k = 1:M-1
            nextS = {};
            for i = 1:size(S, 1)
                slices = sliceList(S{i, 2}, k, ref);
                for j = 1:size(slices, 1)
                    item = {slices{j, 1} * S{i, 1}, slices{j, 2}};
                    nextS = addSlice(item, nextS);
                end
            end
            S = nextS;
        end
        hv = 0;
        for i = 1:size(S, 1)
            p = head(S{i, 2});
            hv = hv + S{i, 1} * abs(p(M) - ref(M));
        end
    else
        sampleNum = 10000;
        maxValue = ref;
        minValue = min(popObj, [], 1);
        samples = minValue + rand(sampleNum, M) .* (maxValue - minValue);
        dominated = false(sampleNum, 1);
        for i = 1:size(popObj, 1)
            dominated = dominated | all(popObj(i, :) <= samples, 2);
        end
        hv = prod(maxValue - minValue) * mean(dominated);
    end
end

function S = sliceList(pl, k, refPoint)
    p = head(pl);
    pl = tail(pl);
    ql = [];
    S = {};
    while ~isempty(pl)
        ql = insertPoint(p, k + 1, ql);
        nextP = head(pl);
        S = addSlice({abs(p(k) - nextP(k)), ql}, S);
        p = nextP;
        pl = tail(pl);
    end
    ql = insertPoint(p, k + 1, ql);
    S = addSlice({abs(p(k) - refPoint(k)), ql}, S);
end

function ql = insertPoint(p, k, pl)
    ql = [];
    hp = head(pl);
    while ~isempty(pl) && hp(k) < p(k)
        ql = [ql; hp]; %#ok<AGROW>
        pl = tail(pl);
        hp = head(pl);
    end
    ql = [ql; p];
    m = length(p);
    while ~isempty(pl)
        q = head(pl);
        flag1 = false;
        flag2 = false;
        for i = k:m
            flag1 = flag1 || p(i) < q(i);
            flag2 = flag2 || p(i) > q(i);
        end
        if ~(flag1 && ~flag2)
            ql = [ql; head(pl)]; %#ok<AGROW>
        end
        pl = tail(pl);
    end
end

function p = head(pl)
    if isempty(pl)
        p = [];
    else
        p = pl(1, :);
    end
end

function ql = tail(pl)
    if size(pl, 1) < 2
        ql = [];
    else
        ql = pl(2:end, :);
    end
end

function S = addSlice(item, S)
    merged = false;
    for k = 1:size(S, 1)
        if isequal(item(1, 2), S(k, 2))
            S(k, 1) = {cell2mat(S(k, 1)) + cell2mat(item(1, 1))};
            merged = true;
            break;
        end
    end
    if ~merged
        S(end + 1, :) = item(1, :);
    end
end

function gc = configDouble(grad1, grad2)
    epsilon = 1e-12;
    norm1 = norm(grad1) + epsilon;
    norm2 = norm(grad2) + epsilon;
    unit1 = grad1 / norm1;
    unit2 = grad2 / norm2;
    cosTheta = dot(grad1, grad2) / (norm1 * norm2);
    ortho2 = grad1 - norm1 * cosTheta * unit2;
    ortho1 = grad2 - norm2 * cosTheta * unit1;
    unitOrtho1 = normalizeVector(ortho1);
    unitOrtho2 = normalizeVector(ortho2);
    denom = dot(unitOrtho1, unit2);
    if abs(denom) < epsilon
        gc = grad1 + grad2;
        return;
    end
    coef1 = dot(unitOrtho2, unit1) / denom;
    bestDirection = coef1 * unitOrtho1 + unitOrtho2;
    gc = rescaleLength(bestDirection, [grad1, grad2]);
end

function u = normalizeVector(v)
    if norm(v) < 1e-12
        u = zeros(size(v));
    else
        u = v / norm(v);
    end
end

function scaled = rescaleLength(targetVector, gradients)
    unitTarget = normalizeVector(targetVector);
    totalProjection = 0;
    for i = 1:size(gradients, 2)
        gi = gradients(:, i);
        totalProjection = totalProjection + norm(gi) * dot(gi, unitTarget) / (norm(gi) * norm(unitTarget) + 1e-12);
    end
    scaled = totalProjection * unitTarget;
end

function W = uniformWeights(N, M)
    raw = -log(max(rand(N, M), realmin));
    W = raw ./ sum(raw, 2);
end

function distances = pairwiseDistances(A, B, upperOnly)
    distances = sqrt(max(0, bsxfun(@plus, sum(A.^2, 2), sum(B.^2, 2)') - 2 * (A * B')));
    if upperOnly
        distances = distances(triu(true(size(distances)), 1));
    end
end

function value = safeRatio(numerator, denominator)
    if isempty(denominator) || abs(denominator) < 1e-12 || isnan(denominator)
        value = 0;
    else
        value = numerator / denominator;
    end
end

function x = rowvec(x)
    x = x(:)';
end
