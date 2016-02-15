local List = require('pl.List')
local tablex = require('pl.tablex')
local path = require('pl.path')
local file = require('pl.file')
require('pl.stringx').import()

COMMANDLINEREAD = nil

local opt = nil

--[[
Parse function also populates:
    opt.savedir: Directory to save current model's results
    opt.hashDir: Directory of hash dictionary for current model
    opt.dataDir: Directory of data files

    opt.subtasks: List of tasks of joint model
    opt.nClass: Mapping of {taskname : nClass} pairs, task name can be found in opt.subtasks
    opt.nhus: Split into list of numbers of hidden units per layer
    opt.AAEmbedSize: Split into list of numbers of embed size of each n-gram
    opt.PSINum: Fixed number of PSI dimensions.
]]
local parse = function(cmd)
    local opt = cmd:parse(arg or {})
    opt.savedir = path.join(opt.resultsDir, opt.dataset,
        cmd:string(opt.name, opt,
            {nosave=true,
             size=true,
             task=true,
             maxEpochs=true,
             dataset=true,
             dataDir=true,
             resultsDir=true,
             name=true,
             noprogressbar=true,
             cuda=true,
             trainingRate=true,
             momentum=true,
             finetune=true,
             kickstart=true,
         })) -- all the non-real training hyper-parameters should be ignored.
    path.mkdir(opt.resultsDir)
    path.mkdir(path.join(opt.resultsDir, opt.dataset))
    path.mkdir(opt.savedir)

    opt.hashDir = path.join(path.abspath(opt.dataDir), opt.dataset, 'hash')
    opt.dataDir = path.join(path.abspath(opt.dataDir), opt.dataset, 'data')

    opt.subtasks = List.split(opt.task, '#')
    opt.nClass = {}
    for subtask in opt.subtasks:iter() do
        local hashfile = path.join(opt.hashDir, subtask..".tag.lst")
        opt.nClass[subtask] = #(assert(file.read(hashfile), "Bad task specification, check data directory for list of tasks"):splitlines())
    end

    opt.nhus = tablex.map(tonumber, opt.nhus:split(','))
    opt.AAEmbedSize = tablex.map(tonumber, opt.AAEmbedSize:split(','))
    -- Fixed Parameters
    opt.PSINum = 20

    -- assert(opt.modelparams ~= '', "Please input some parameters for your model with -modelparams!")

    return opt
end

local set = function()
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text()
    cmd:text('Options:')
    cmd:text()
    cmd:option('-CVfold', 0, 'Cross validation fold used. 0 will use a train-valid-test model with a 60-20-20 split')
    cmd:option('-task',"dssp.lab#absolute.lab#ssp.lab#sa-relative.lab",
        'Task name. Specific to this joint model, corresponds to' ..
        'the labeling task we want to perform') -- These are all of the multitask tasks

    cmd:option('-finetune', false, "Replace the last layers and finetune the model on the new tasks. Will fail if we can't find existing model")
    cmd:option('-kickstart', "", "Replace the last layers and finetune the model on the new tasks. Will fail if we can't find existing model")
    cmd:option('-size', 0, 'How many samples to use. 0 should load all samples.')

    cmd:option('-AAEmbedSize', '15,0,0,0',
        'Size of alphabet embeddings to use. Format is a,b,c,d, where ' ..
        'a is the size of the unigram embedding, b is bigram, etc.' )

    cmd:option('-model', 'conv', 'mlp | conv')
    cmd:option('-nonlinearity', 'relu', 'type of nonlinearity function to use: tanh | relu | prelu')
    cmd:option('-loss', 'nll', 'type of loss function to minimize: nll')
    cmd:option('-nhus', '1024,1024,1024', 'Number of hidden units in each layer')
    cmd:option('-modelparams', '5,5,3', 'Convolution kernel sizes of conv, '..
                                    'input model to test for test_model.lua')
    cmd:option('-pools', '2,2,2', 'maxpooling sizes')
    cmd:option('-dropout', 0.3, 'dropout rate.')
    cmd:option('-indropout', 0.1, 'dropout rate for input.')

    cmd:option('-optimization', 'SGD', 'optimization method: SGD | CG | LBFGS')
    cmd:option('-trainingRate', 1e-3, 'learning rate at t=0')
    cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-weightDecay', 1e-7, 'weight decay (SGD only)')
    cmd:option('-momentum', 0.9, 'momentum (SGD only)')

    cmd:option('-cuda', false, 'Use GPU')

    cmd:option('-maxEpochs', 0, 'How many epochs to run for, where 0 is infinite')

    cmd:option('-noprogressbar', false, 'Whether to display a progress bar')
    cmd:option('-nosave', false, 'Whether to save the model')

    cmd:option('-dataset', "4Protein", 'Dataset name, corresponds to the folder name in dataDir')
    cmd:option('-dataDir', "./data", 'The data home location')
    cmd:option('-resultsDir', "./results", 'The data home location')
    cmd:option('-name', "", 'Optionally, give a name for this model.')

    opt = parse(cmd)
end

-- Let's not read the command line multiple times
if not COMMANDLINEREAD then
    set()
end

return opt
