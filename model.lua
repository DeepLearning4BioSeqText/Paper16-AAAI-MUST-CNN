require 'nn'
local nn_extras = require('DeepUtils/nn_extras')
require('pl.stringx').import()
local file = require('pl.file')
local List = require('pl.List')

-- Parse cmd line
local opt = require('cmdlineargs')

local function optget(tbl, option, message)
    local ret = tbl[option]
    assert(ret ~= nil, message)
    return ret
end

-- Parse nonlinearity
local nltbl = {
    ['tanh'] = nn.Tanh,
    ['relu'] = nn.ReLU,
    ['prelu'] = nn.PReLU,
}
local nonlinearity = optget(nltbl, opt.nonlinearity, "Invalid nonlinearity of "..opt.nonlinearity)


-- Let's assume a length M sequence
-- N is AA embed sizes + psi sizes

local function create_common_model(seq)
    local model = seq
    local prl = nn.ParallelTable()

    opt.processed_dim = 0

    for i=1,4 do if opt.AAEmbedSize[i] > 0 then
        local aadict = file.read(path.join(opt.hashDir, string.format("aa%d.lst", i)))
        local aaxsize = #(aadict:splitlines())

        local wordlookup = nn.LookupTable(aaxsize, opt.AAEmbedSize[i])
        prl:add(wordlookup)

        opt.processed_dim = opt.processed_dim + opt.AAEmbedSize[i]
    end end

    for i=1,opt.PSINum do
        prl:add(nn.UpDim(nn.UpDim.END))
        opt.processed_dim = opt.processed_dim + 1
    end

    -- Table of N M-lengthed sequences
    model:add(prl)
    model:add(nn.JoinTable(2))
    -- MxN matrix
    if opt.indropout > 0 then model:add(nn.Dropout(opt.indropout)) end

    return model
end

-- Convolution Models
local function create_conv_base_model(seq)
    local params = opt.modelparams
    local pools = opt.pools
    local conv_kernels = tablex.map(tonumber, params:split(','))
    local pools = tablex.map(tonumber, pools:split(','))
    assert(#pools == #conv_kernels, "num pools must equal num conv_kernels")
    local model = seq
    model:add(nn.UpDim())

    opt.nhus[0] = opt.processed_dim
    for i,_ in ipairs(opt.nhus) do
        local kern = conv_kernels[i] or 1
        local pool = pools[i] or 1
        local padding = math.floor(kern/2)
        if padding > 0 then
            local concat = nn.ConcatTable()
            for j = 0,pool-1 do
                concat:add(nn.SpatialZeroPadding(0,0,padding-j,kern-padding-1+j+pool))
            end
            model:add(concat)
            model:add(nn.JoinTable(1))
        end
        model:add(nn.TemporalConvolution(opt.nhus[i-1], opt.nhus[i], kern))
        model:add(nonlinearity())
        if pool > 1 then
            model:add(nn.TemporalMaxPooling(pool))
        end
        if opt.dropout > 0 then model:add(nn.Dropout(opt.dropout)) end
    end

    model:add(nn.TemporalZip())
    return model
end

local function create_conv_sub_model(nClass)
	return nn.Sequential():add(nn.TemporalConvolution(opt.nhus[#opt.nhus], nClass, 1))
end

local function create_conv_model()
    local model = nn.Sequential()
    create_common_model(model)
    create_conv_base_model(model)
    local sub_mlp_concat = nn.ConcatTable()
    for task in opt.subtasks:iter() do
        sub_mlp_concat:add(create_conv_sub_model(opt.nClass[task]))
    end
    model:add(sub_mlp_concat)

    return model
end


-- Criterions
local function nll_criterion(model)
    local concat_table = model:get(model:size())
    local criterion = nn.MultiCriterionTable()
    for i,task in ipairs(opt.subtasks) do
        criterion:add( nn.ClassNLLCriterion() )

        concat_table:get(i):add(nn.LogSoftMax())
    end
    return criterion
end


local criterion_choice = {
    ['nll'] = nll_criterion,
}

local function getModel()
    local model = create_conv_model()
    local criterion = criterion_choice[opt.loss](model)

    return model, criterion
end

local function finetune_model(model)
    local sub_model_choice = {
        ['conv'] = create_conv_sub_model,
    }

    local sub_mlp_concat = nn.ConcatTable()
    for task in opt.subtasks:iter() do
        sub_mlp_concat:add(sub_model_choice[opt.model](opt.nClass[task]))
    end
    model.modules[1].modules[1]._input = torch.LongTensor()
    model.modules[model:size()] = sub_mlp_concat
    model.output = sub_mlp_concat.output

    criterion = criterion_choice[opt.loss](model)

    return model, criterion
end

return {
    getModel=getModel,
    finetune_model=finetune_model
}
