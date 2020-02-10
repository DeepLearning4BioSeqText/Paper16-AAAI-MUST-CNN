local xlua = require('xlua')
require('nn')
require('optim')
local tablex = require('pl.tablex')

local model_cache = nil
local parameters = nil
local gradParameters = nil

local function train(model, criterion, data, labels, params)
    if model ~= model_cache then
        model_cache = model
        parameters,gradParameters = model:getParameters()
    end
    local trainopt = {
        optimMethod = optim.sgd,
        batchSize = 1,
        progressbar = true,
        doAfterEveryExample = function() end,
        optimState = {},
    }

    params = params or {}
    for k,v in pairs(params) do
        assert(trainopt[k], "Invalid Option")
        trainopt[k] = v
    end
    local shuffle = torch.IntTensor(data:size(1)):randperm(data:size(1))

    -- Set training mode
    model:training()

    local error_accum = 0

    for t = 1,data:size(1),trainopt.batchSize do
        -- disp progress
        if trainopt.progressbar then
            xlua.progress(t, data:size(1))
        end

        -- create mini batch
        local inputs = {}
        local targets = {}
        for i = t,math.min(t+trainopt.batchSize-1,data:size(1)) do
            -- load new sample
            local input = data[shuffle[i]]
            local target = labels[shuffle[i]]
            table.insert(inputs, input)
            table.insert(targets, target)
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0

            -- evaluate function for complete mini batch
            for i = 1,#inputs do
                -- estimate f
                local output = model:forward(inputs[i])

                --clip extra output from shift and stitch
                output = tablex.map(function(t) return t[{{1,targets[i][1]:size(1)},{1,t:size(2)}}] end, output)

                local err = criterion:forward(output, targets[i])
                f = f + err

                -- estimate df/dW
                local df_do = criterion:backward(output, targets[i])
                model:backward(inputs[i], df_do)

                trainopt.doAfterEveryExample(output, targets[i], err)
            end

            -- normalize gradients and f(X)
            gradParameters:div(#inputs)
            f = f/#inputs

            error_accum = 0.95*error_accum + 0.05*f -- Exponential running average of error
            -- return f and df/dX
            return f,gradParameters
        end

        -- optimize on current mini-batch
        trainopt.optimMethod(feval, parameters, trainopt.optimState)
        if not (error_accum > -math.huge and error_accum < math.huge) then --Error is infinite! We exit immediately
            return error_accum
        end
    end

    return error_accum
end

-- test function
local function test(model, criterion, data, labels, params)
    local testopt = {
        progressbar = true,
        doAfterEveryExample = function() end,
    }

    params = params or {}
    for k,v in pairs(params) do
        assert(testopt[k], "Invalid Option")
        testopt[k] = v
    end
    model:evaluate()

    local f = 0

    -- test over test data
    for t = 1,data:size(1) do
        -- disp progress
        if testopt.progressbar then
            xlua.progress(t, data:size(1))
        end

        -- get new sample
        local input = data[t]
        local target = labels[t]

        -- test sample
        local pred = model:forward(input)
        pred = tablex.map(function(t) return t[{{1,target[1]:size(1)},{1,t:size(2)}}] end, pred)
        local err = criterion:forward(pred, target)
        f = f + err

        testopt.doAfterEveryExample(pred, target)
    end

    return f/data:size(1)
end

local Trainer = {
    train=train,
    test=test,
}

return Trainer
