require 'gnuplot'
require 'nn'
require 'optim'
local func = require('pl.func')
local path = require('pl.path')


local function acc(field, p)
    print(p)
    local log = torch.load(path.join(p, "log.dat"))
    local s = tablex.size(log[#log].matrix)

    return tablex.imap(function(v)
        return tablex.reduce('+', 
            tablex.values(tablex.map(function(taskconf)
                    return taskconf.totalValid 
                end,
                v[field]
            )
        )
    ) / s end, tablex.filter(log, function(v) return v end))
end

local validacc = func.bind1(acc, "validationmatrix")
local trainacc = func.bind1(acc, "matrix")

local function _V(f, p)
    local data = f(p)
    local x = tablex.keys(data)
    local y = tablex.index_by(data, x)
    return torch.Tensor(x), torch.Tensor(y)
end

local va = func.bind1(_V, validacc)
local ta = func.bind1(_V, trainacc)
local nt = {}
nt[va] = "valid"
nt[ta] = "train"

function plotn(savepath, title, names, paths, funcs)
    local plotter = {}
    local xmin = 0
    local xmax = 0
    for i,name in ipairs(names) do
        for _,f in pairs(funcs) do
            local x,y = f(paths[i])
            table.insert(plotter, {name.."-"..nt[f], x, y, '-'})
            xmin = math.min(xmin, x:min())
            xmax = math.max(xmax, x:max())
        end
    end
    table.insert(plotter, 1, {'plosone baseline', torch.range(xmin, xmax), torch.Tensor(xmax-xmin+1):fill(.784), 'lines pt 2 lw 1'})
    gnuplot.pngfigure(savepath)
    gnuplot.movelegend("right", "bottom")
    gnuplot.title(title)
    gnuplot.plot(plotter)
    gnuplot.xlabel('epochs')
    gnuplot.ylabel('Average % accuracy')
    gnuplot.plotflush()
end

local root = "../results/4Protein/"
local p = path.join(root, "exp6.19fixed,dropout=0.2,indropout=0.1,model=conv,modelparams=9,7,3,3,3,nhus=512,512,512,512,nonlinearity=relu")
plotn("graphs/train_vs_valid.png", "Train vs Validation on best model", {''}, {p}, {va, ta})

local p1 = path.join(root, "exp6.19fixed,dropout=0.2,indropout=0.1,model=conv,modelparams=9,7,3,3,3,nhus=512,512,512,512,nonlinearity=relu")
local p2 = path.join(root, "exp6.19fixed,model=conv,modelparams=9,7,3,3,3,nhus=512,512,512,512,nonlinearity=relu")
plotn("graphs/dropout.png", "Dropout vs no Dropout on best model", {'dropout', 'no-dropout'}, {p1, p2}, {va, ta})

local p1 = path.join(root, "exp6.19fixed,dropout=0.2,indropout=0.1,model=conv,modelparams=9,7,3,3,3,nhus=512,512,512,512")
local p2 = path.join(root, "exp6.19fixed,dropout=0.2,indropout=0.1,model=conv,modelparams=9,7,3,3,3,nhus=512,512,512,512,nonlinearity=relu")
local p3 = path.join(root, "exp6.19fixed,dropout=0.2,indropout=0.1,model=conv,modelparams=9,7,3,3,3,nhus=512,512,512,512,nonlinearity=prelu")
plotn("graphs/nonlinearities.png", "different nonlinearity on best model", {"tanh", "relu", "prelu"}, {p1, p2, p3}, {va})

local p1 = path.join(root, "exp6.19fixed,dropout=0.2,indropout=0.1,model=conv,modelparams=9,7,3,3,3,nhus=512,512,nonlinearity=relu")
local p2 = path.join(root, "exp6.19fixed,dropout=0.2,indropout=0.1,model=conv,modelparams=9,7,3,3,3,nhus=512,512,512,nonlinearity=relu")
local p3 = path.join(root, "exp6.19fixed,dropout=0.2,indropout=0.1,model=conv,modelparams=9,7,3,3,3,nhus=512,512,512,512,nonlinearity=relu")
local p4 = path.join(root, "exp6.19fixed,dropout=0.5,indropout=0.2,model=conv,modelparams=9,7,3,3,3,nhus=512,512,512,512,512,nonlinearity=relu")
plotn("graphs/layers.png", "different number of layers on best model", {"2 layers", "3 layers", "4 layers", "5 layers"}, {p1, p2, p3, p4}, {va, ta})










