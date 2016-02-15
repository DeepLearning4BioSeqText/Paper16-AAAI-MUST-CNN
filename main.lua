local nn = require('nn')
local nn_extras = require('DeepUtils/nn_extras')
local utils = require('pl.utils')
local seq = require('pl.seq')
local path = require('pl.path')
local tablex = require('pl.tablex')
local modellib = require('model')
--local ProFi = require('DeepUtils/ProFi')

print('==> switching to floats')
torch.setdefaulttensortype('torch.FloatTensor')

local function loadModel()
    -- Parse cmd line
    local opt = require('cmdlineargs')

    if opt.cuda then
        --cunn,ok = pcall(require, 'fbcunn')
        if not ok then
            ok,cunn = pcall(require, 'cunn')
        else
            nn.TemporalMaxPooling = nn.TemporalKMaxPooling
            nn.TemporalConvoluion = nn.TemporalConvolutionFB
        end
        if not ok then error("No CUDA torch modules installed.") end
        print('==> switching to cuda')
        torch.setdefaulttensortype('torch.CudaTensor')
    end

    local state = {}
    local log = {}
    local savepath = path.join(opt.savedir, "state.dat")
    local modelbasepath = path.join(opt.savedir, "model.net.epoch.")
    local logpath = path.join(opt.savedir, "log.dat")
    local not_saved = { 'size', 'nosave', 'maxEpochs', 'noprogressbar', 'trainingRate', 'momentum', 'finetune' }


    if opt.finetune then
        local finetunedir = path.join(opt.savedir, "finetune_"..opt.task)
        local finetunepath = path.join(opt.savedir, "finetune_"..opt.task, "state.dat")
        path.mkdir(finetunedir)
        print(savepath)
        if not path.exists(savepath) then error("Cannot find model to finetune") end


        if path.exists(finetunepath) then
            modelbasepath = path.join(finetunedir, "model.net.epoch.")
            logpath = path.join(finetunedir, "log.dat")
            print("Restoring the previous model")
            savepath = path.join(finetunedir, "state.dat")
            state = torch.load(savepath)
            log = torch.load(logpath)
            -- Config that shouldn't be saved
            tablex.map(function (v) state.opt[v] = opt[v] end, not_saved)

            state.epoch = state.epoch + 1

            for k,v in pairs(opt) do opt[k] = nil end
            for k,v in pairs(state.opt) do opt[k] = state.opt[k] end
            opt.subtasks = List.split(opt.task, '#')

        else
            state = torch.load(savepath)
            log = torch.load(logpath)
            tablex.map(function (v) state.opt[v] = opt[v] end, not_saved)
            state.opt.task = opt.task

            local best_model = tablex.imap(function(v)
                if not v then return 0 end
                local v =  tablex.reduce('+',
                    tablex.values(tablex.map(function(taskconf)
                        return taskconf.totalValid end,
                        v.validationmatrix
                    )
                ))
                if v ~= v then return 0 else return v end
            end, log)
            local s,bmi = torch.DoubleTensor(best_model):max(1)
            bmi = bmi[1]
            state.model = torch.load(modelbasepath .. bmi)
            print("Finetuning from model at epoch " .. bmi)

            state.model, state.criterion = modellib.finetune_model(state.model)

            state.epoch = 1
            for k,v in pairs(opt) do opt[k] = nil end
            for k,v in pairs(state.opt) do opt[k] = state.opt[k] end
            opt.subtasks = List.split(opt.task, '#')

            -- have to reset the training state
            state.optimState = {
                -- SGD
                learningRate = opt.trainingRate,
                weightDecay = opt.weightDecay,
                momentum = opt.momentum,
                learningRateDecay = 1e-7,
                -- CG and LBFGS
                maxIter = opt.maxIter,
                -- Just LBFGS
                nCorrection = 10,
            }

            modelbasepath = path.join(finetunedir, "model.net.epoch.")
            logpath = path.join(finetunedir, "log.dat")
            savepath = path.join(finetunedir, "state.dat")
            log = {}
        end

        opt.CVfold = 1 -- Finetune also on validation set

    elseif path.exists(savepath) then
        print("Restoring the previous model")
        state = torch.load(savepath)
        log = torch.load(logpath)
        -- Config that shouldn't be saved
        tablex.map(function (v) state.opt[v] = opt[v] end, not_saved)

        state.epoch = state.epoch + 1

        for k,v in pairs(opt) do opt[k] = nil end
        for k,v in pairs(state.opt) do opt[k] = state.opt[k] end
        opt.subtasks = List.split(opt.task, '#')
    else
        print("No previous model found. Creating new model")
        local model,crit = modellib.getModel()

        state.epoch = 1
        state.optimState = {
            -- SGD
            learningRate = opt.trainingRate,
            weightDecay = opt.weightDecay,
            momentum = opt.momentum,
            learningRateDecay = 1e-7,
            -- CG and LBFGS
            maxIter = opt.maxIter,
            -- Just LBFGS
            nCorrection = 10,
        }
        state.optimMethod = optim.sgd
        state.opt = opt
        state.model = model
        state.criterion = crit

        if opt.kickstart ~= "" then
            local p,_ = state.model:getParameters()
            local kickstarter = torch.load(opt.kickstart)
            local op,_ = kickstarter:getParameters()
            p:copy(op)
        end
    end
    opt.savepath = savepath
    opt.modelbasepath = modelbasepath
    opt.logpath = logpath
    if opt.cuda then
        state.model:cuda()
        state.criterion:cuda()
    end

    return state, opt, log
end


local function main()
    local state, opt, log = loadModel()

    print("====== ARGUMENTS =======")
    print(opt)
    print("====== MODEL ======")
    print(state.model)
    print("====== CRITERION ======")
    print(state.criterion)

    -- data.{train, test}.{inputs, labels}
    print("Requiring data...")
    local data = require('data')

    local confusion = nn_extras.JointConfusionMatrix.new(opt.subtasks, opt.nClass)
    local function recordData(output, target, err)
        confusion:add(output, target)
    end

    local trainopt = {
        optimMethod = state.optimMethod,
        progressbar = not opt.noprogressbar,
        doAfterEveryExample = recordData,
        optimState = state.optimState,
    }
    local testopt = {
        progressbar = not opt.noprogressbar,
        doAfterEveryExample = recordData,
    }

    -- MAIN LOOP
    --ProFi:start()
    while state.epoch <= opt.maxEpochs or opt.maxEpochs == 0 do
        log[state.epoch] = {}

        local time = torch.tic()
        utils.printf("Epoch %d: training...\n", state.epoch)

        local err = nn_extras.Trainer.train(state.model, state.criterion, data.train.inputs, data.train.labels, trainopt)

        time = torch.toc(time)
        utils.printf("Finished epoch in %.3f seconds\n", time)

        if not (err > -math.huge and err < math.huge) then --Error is infinite! We exit immediately
            print("model diverged :(")
            return 0
        end

        print(confusion:__tostring__())
        log[state.epoch].matrix = confusion.matricies
        log[state.epoch].time = time
        confusion:zero()

        utils.printf("Epoch %d: testing on validation set...\n", state.epoch)
        local time = torch.tic()
        nn_extras.Trainer.test(state.model, state.criterion, data.valid.inputs, data.valid.labels, testopt)
        time = torch.toc(time)

        print(confusion:__tostring__())
        log[state.epoch].validationmatrix = confusion.matricies
        log[state.epoch].validationtime = time
        confusion:zero()

        -- Log data
        if not opt.nosave then
            torch.save(opt.savepath, state)
            torch.save(opt.logpath, log)
            torch.save(opt.modelbasepath .. state.epoch, state.model)
        end

        -- Reset
        state.epoch = state.epoch + 1
        collectgarbage()
    end
    --ProFi:stop()
    --ProFi:writeReport('profile.log')

    local best_model = tablex.imap(function(v)
        if not v then return 0 end
        return tablex.reduce('+',
            tablex.values(tablex.map(function(taskconf)
                return taskconf.totalValid end,
                v.validationmatrix
            )
        )
    ) end, log)
    local s,bmi = torch.DoubleTensor(best_model):max(1)
    bmi = bmi[1]
    if opt.finetune then bmi = #best_model end
    if not nosave then
        state.model = torch.load(opt.modelbasepath .. bmi)
    end
    if opt.cuda then state.model:cuda() else state.model:float() end
    utils.printf("Testing model at epoch %s...\n", bmi)

    local time = torch.tic()
    nn_extras.Trainer.test(state.model, state.criterion, data.test.inputs, data.test.labels, testopt)
    time = torch.toc(time)
    print(confusion:__tostring__())

    log["test"] = {}
    log["test"].matrix = confusion.matricies
    log["test"].time = time
    log["test"].epoch = bmi
    if not opt.nosave then
        torch.save(opt.logpath, log)
    end

    return s[1] / #opt.subtasks
end

print(main())
