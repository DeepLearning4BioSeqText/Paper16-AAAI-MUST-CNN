local UpDim, parent = torch.class('nn.UpDim', 'nn.Module')

-- Turns a axbxc... tensor into a 1xaxbxc... tensor
UpDim.END = 0
UpDim.FRONT = 1

function UpDim:__init(position)
    parent.__init(self)
    if position == UpDim.FRONT then
        self.front = true
    elseif position == UpDim.END then
    self.front = false
else
    self.front = UpDim.END
end
end

function UpDim:updateOutput(input)
    if self.front then
        self.output = input:reshape(1,unpack(input:size():totable()))
    else
        local dims = input:size():totable()
        table.insert(dims,1)
        self.output = input:reshape(unpack(dims))
    end
    return self.output
end

function UpDim:updateGradInput(input, gradOutput)
    if self.front then
        self.gradInput = gradOutput:select(1,1)
    else
        self.gradInput = gradOutput:select(input:dim()+1,1)
    end
    return self.gradInput
end

