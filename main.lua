local Trainer = require('Trainer')
local JointConfusionMatrix = require('JointConfusionMatrix')
local modellib = require('model')
local path = require('pl.path')
local file = require('pl.file')
require('pl.stringx').import()

cmd = torch.CmdLine()
cmd:text('MustCNN training')
cmd:text()
cmd:text('Options')
cmd:option('-cuda', false,'CUDA support')
cmd:option('-gpu', 1,'GPU device number')
cmd:option('-task',"absolute.lab", 'Task name') 
cmd:option('-embedsize',15,"Size of alphabet embeddings to use")
cmd:option('-kernelsize',5,"Convolution kernel size")
cmd:option('-hiddenunit',1024,"Number of hidden units")
cmd:option('-poolingsize',2,"Max pooling size")
cmd:option('-nonlinearity','relu', 'type of nonlinearity function to use: tanh | relu | prelu')
cmd:text()
params = cmd:parse(arg)

cuda = params.cuda
gpudevicenum = params.gpu
taskname = params.task

params.dataset = "./data/4Protein"
params.hashdir = path.join(params.dataset,'hash')
params.datadir = path.join(params.dataset,'data')

local hashfile = path.join(params.hashdir, taskname..".tag.lst")
params.nclass = #(file.read(hashfile):splitlines())

print(params)
torch.setdefaulttensortype('torch.FloatTensor')



local function main()
	local data = require('loaddata')
	print('finish loading data')
    if cuda then
        require 'cutorch'
        cutorch.setDevice(gpudevicenum)
        ok,cunn = pcall(require, 'cunn')
        if not ok then error("No CUDA torch modules installed.") end
        print('==> switching to cuda')
        torch.setdefaulttensortype('torch.CudaTensor')
    end	

    local model, crit = modellib.getmodel(params)
	print(model)
	print(crit)
    if cuda then
        model:cuda()
        crit:cuda()
    end
    tasklabels = {}
    tasklabels[taskname] = nclass

    local confusion = JointConfusionMatrix.new({taskname},tasklabels)

    local function recordData(output, target, err)
        confusion:add(output, target)
    end
    -- local ConfusionMatrix = torch.class('optim.ConfusionMatrix')

    local trainopt = {
        optimMethod = optim.sgd,
        progressbar = true,
        doAfterEveryExample = recordData,
        optimState = {
                -- SGD
                learningRate = 1e-3,
                weightDecay = 1e-7,
                momentum = 0.9,
                learningRateDecay = 1e-7,
            },
    }
    local testopt = {
        progressbar = true,
        doAfterEveryExample = recordData,
    }

    epoch = 0
    local log={}
    while true do
    	log[epoch] = {}
        local time = torch.tic()
        utils.printf("Epoch %d: training...\n", epoch)

        local err = Trainer.train(model, crit, data.train.inputs, data.train.labels, trainopt)

        time = torch.toc(time)
        utils.printf("Finished epoch in %.3f seconds\n", time)
        if not (err > -math.huge and err < math.huge) then --Error is infinite! We exit immediately
            print("model diverged :(")
            return 0
        end

        print(confusion:__tostring__())
        log[epoch].matrix = confusion.matricies
        log[epoch].time = time
        confusion:zero() 

        local time = torch.tic()
        utils.printf("Epoch %d: testing on validation set...\n", epoch)
        Trainer.test(model, crit, data.valid.inputs, data.valid.labels, testopt)
		time = torch.toc(time)

        print(confusion:__tostring__())
        log[epoch].validationmatrix = confusion.matricies
        log[epoch].validationtime = time
        confusion:zero()

        epoch = epoch + 1
        
        torch.save("log", log)
        torch.save("model", model)
        collectgarbage()
    end
	-- print(data)
end

main()