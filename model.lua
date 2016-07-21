require 'nn'

require("component")
require('pl.stringx').import()
local file = require('pl.file')
local List = require('pl.List')

-- Parse nonlinearity
local nltbl = {
    ['tanh'] = nn.Tanh,
    ['relu'] = nn.ReLU,
    ['prelu'] = nn.PReLU,
}

local function create_common_model(seq)
    local model = seq
    local aadict = file.read(path.join(hashdir, "aa1.lst"))
    local aaxsize = #(aadict:splitlines())
    model:add(nn.LookupTable(aaxsize, embedsize))
    model:add(nn.Dropout(0.1))

    return model
end

-- Convolution Models
local function create_conv_base_model(seq)

    local model = seq
    model:add(nn.UpDim())

    local padding = math.floor(kernelsize/2)
    if padding > 0 then
        local concat = nn.ConcatTable()
        for j = 0,poolingsize-1 do
            concat:add(nn.SpatialZeroPadding(0,0,padding-j,kernelsize-padding-1+j+poolingsize))
        end
        model:add(concat)
        model:add(nn.JoinTable(1))
    end
    model:add(nn.TemporalConvolution(embedsize, hiddenunit, kernelsize))
    model:add(nonlinearity())
    if poolingsize > 1 then
        model:add(nn.TemporalMaxPooling(poolingsize))
    end
    model:add(nn.Dropout(0.3))
    -- end

    model:add(nn.TemporalZip())
    return model
end



local function create_conv_model()
    local model = nn.Sequential()
    create_common_model(model)
    create_conv_base_model(model)
    model:add(nn.TemporalConvolution(hiddenunit, nclass, 1))
    return model
end


local function getmodel(params)
    hashdir = params.hashdir
    nclass = params.nclass
    embedsize = params.embedsize
    kernelsize = params.kernelsize
    hiddenunit = params.hiddenunit
    poolingsize = params.poolingsize

    nonlinearity = nltbl[params.nonlinearity]

    local model = create_conv_model()
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    print(criterion)
    return model, criterion
end

return {
    getmodel = getmodel
}


