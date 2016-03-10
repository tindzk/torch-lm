require "rnn"
local fun = require "fun"

local CNN = {}

function CNN.getParallelConvolution(inputSize, filterMinWidth, filterMaxWidth)
  local function getConvolutionModule(inputSize, filterWidth)
    local conv = nn.Sequential()
    conv:add(nn.TemporalConvolution(inputSize, 1, filterWidth)) -- convolution over time
    conv:add(nn.TemporalMaxPooling(1))
    return conv
  end

  local convModule = nn.ParallelTable()
  fun.each(
    function (width)
      convModule:add(getConvolutionModule(inputSize + (width - 1) * 2, width))
    end,
    fun.range(filterMinWidth, filterMaxWidth)
  )

  local convOutputLength = torch.range(filterMinWidth, filterMaxWidth):sum()

  local joinModule = nn.Sequential()
  joinModule:add(nn.JoinTable(1))
  joinModule:add(nn.Reshape(convOutputLength))

  local net = nn.Sequential()
  net:add(nn.Sequencer(convModule))
  net:add(nn.Sequencer(joinModule))

  return net
end

function CNN.addPadding(xTable, filterMinWidth, filterMaxWidth)
  return fun.map(
    function (tensor)
      return fun.map(
        function (width)
          if width > 1 then
            local zeros = torch.zeros(tensor:size(1), width - 1)
            local res = torch.cat(zeros, tensor, 2)
            return torch.cat(res, zeros, 2)
          else
            return tensor
          end
        end,
        fun.range(filterMinWidth, filterMaxWidth)
      ):totable()
    end, xTable):totable()
end

function CNN.test()
  -- CNN hyperparameters
  local filterMinWidth = 1
  local filterMaxWidth = 10
  local inputSize = 15

  -- Dummy data similar to our real dataset.
  local dataset = {
    torch.rand(10, inputSize),
    torch.rand(10, inputSize),
    torch.rand(10, inputSize),
    torch.rand(10, inputSize)
  }

  local input = CNN.addPadding(dataset, filterMinWidth, filterMaxWidth)
  local net = CNN.getParallelConvolution(inputSize, filterMinWidth, filterMaxWidth)
  local result = net:forward(input)
  print("test passed:", result[4]:size(1) == 55)
end

return CNN