require 'optim'
local class = require('class')
local tablex = require('pl.tablex')

local JointConfusionMatrix = torch.class("JointConfusionMatrix")

function JointConfusionMatrix:__init(taskOrder, taskNum)
    self.task_list = taskOrder
    self.matricies = {}
    for task,nClasses in pairs(taskNum) do
        self.matricies[task] = optim.ConfusionMatrix(nClasses)
    end
    return self
end

-- Accepts a table of output labels for each task and target labels,
-- or a table of a sequence of such labels.
function JointConfusionMatrix:add(output, targets)
    for task_id,task in ipairs(self.task_list) do
        if output[1]:dim() == 1 then
            if targets:size(1) == 1 then 
                self.matricies[task]:add(output, targets)
            else
                self.matricies[task]:batchAdd(output, targets)
            end
        elseif output[1]:dim() == 2 then
            for i=1,targets[task_id]:size(1) do
                self.matricies[task]:add(output[task_id][i], targets[task_id][i])
            end
        end
    end
end

function JointConfusionMatrix:zero()
    self.matricies = tablex.map(function (v) return optim.ConfusionMatrix(v.nclasses) end, self.matricies)
end

function JointConfusionMatrix:legend_table(with_key)
    local legend_table = {}
    for _, task in pairs(self.task_list) do
        if with_key then
            legend_table[task] = '-'
        else
            table.insert(legend_table, '-')
        end
    end
    return legend_table
end

function JointConfusionMatrix:total_table(job_type, with_key)
    local total_table = {}
    local total_table_no_key = {}
    for _, task in pairs(self.task_list) do
        total_table[job_type..'.'..task] = self[task].totalValid * 100
        table.insert(total_table_no_key, self[task].totalValid * 100)
    end
    return with_key and total_table or total_table_no_key
end


function JointConfusionMatrix:__tostring__()
    str = 'ConfusionMatrix for joint training: \n\n'
    for _,task in pairs(self.task_list) do
        str = str .. "Task name: ".. task .. '\n'
                  .. self.matricies[task]:__tostring__() .. '\n'
    end
    return str
end

return JointConfusionMatrix
