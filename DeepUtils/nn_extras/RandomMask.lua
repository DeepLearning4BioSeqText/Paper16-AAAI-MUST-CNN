local RandomMask, Parent = torch.class('nn.RandomMask', 'nn.Module')

function RandomMask:__init(size, p)
    Parent.__init(self)
    self.p = p or 0.5
    self.train = true
    -- version 2 scales output during training instead of evaluation
    if self.p >= 1 or self.p < 0 then
        error('<RandomMask> illegal percentage, must be 0 <= p < 1')
    end
    self.noise = torch.Tensor(size)
    self.noise:bernoulli(1-self.p)
    self.noise:div(1-self.p)
end

function RandomMask:updateOutput(input)
    self.output:resizeAs(input):copy(input)
    if not self.train then
        if self.output:dim() == 2 then
            -- Batch
            self.output:cmul(self.noise:repeatTensor(self.output:size(1), 1))
        else
            self.output:cmul(self.noise)
        end
    end
    return self.output
end

function RandomMask:updateGradInput(input, gradOutput)
    if self.train then
        self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    else
        error('backprop only defined while training')
    end
    return self.gradInput
end

function RandomMask:setp(p)
    self.p = p
    self:reset()
end

function RandomMask:reset(seed)
    if seed then torch.manualSeed(seed) end
    self.noise:bernoulli(1-self.p)
    self.noise:div(1-self.p)
    torch.seed()
end

function RandomMask:__tostring__()
    return string.format('%s(%f)', torch.type(self), self.p)
end
