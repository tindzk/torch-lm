require "rnn"
local fun = require "fun"

local CNN = {}

function CNN.convolutionModule(inputFrameSize, outputFrameSize, kernelWidth, backend)
  local conv = nn.Sequential()
  if backend == "cl" then
    conv:add(nn.TemporalConvolution2(inputFrameSize, outputFrameSize, kernelWidth))
    conv:add(nn.SpatialMaxPooling(outputFrameSize, 1))
  else
    conv:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kernelWidth))
    conv:add(nn.TemporalMaxPooling(outputFrameSize))
  end
  return conv
end

function CNN.convolutionTable(convType)
  if convType == "wide" then
    return nn.ParallelTable()
  elseif convType == "narrow" then
    return nn.ConcatTable()
  end
end

-- TODO Create separate functions: wideConvolution, narrowConvolution
function CNN.parallelConvolution(convType, alphabetLen, charEmbeddingLen, inputSize, filterMinWidth, filterMaxWidth, backend)
  -- FIXME: what the value for outputFrameSize shoud be is unclear for me
  local outputFrameSize = 1

  local convModule = CNN.convolutionTable(convType)
  fun.each(
    function(width)
      if convType == "wide" then
        convModule:add(CNN.convolutionModule(inputSize + (width - 1) * 2, outputFrameSize, width, backend))
      elseif convType == "narrow" then
        convModule:add(CNN.convolutionModule(charEmbeddingLen, outputFrameSize, width, backend))
      end
    end,
    fun.range(filterMinWidth, filterMaxWidth)
  )

  local joinModule = nn.Sequential()

  if convType == "wide" then
    joinModule:add(nn.JoinTable(1))
    -- FIXME: wrong calculations here
    local convOutputLength = torch.range(filterMinWidth, filterMaxWidth):sum()
    joinModule:add(nn.Reshape(convOutputLength))

  elseif convType == "narrow" then
    joinModule:add(nn.JoinTable(2))
    local convOutputLength = torch.range(inputSize - (filterMaxWidth - filterMinWidth), inputSize):sum()
    joinModule:add(nn.Reshape(convOutputLength))
  end

  local net = nn.Sequential()
  if convType == "narrow" then
    -- character embedding
    net:add(nn.Sequencer(nn.LookupTable(alphabetLen, charEmbeddingLen)))
  elseif convType == "wide" then
    -- TODO: embedding for wide convolution.
  end

  -- convolution itself
  net:add(nn.Sequencer(convModule))
  -- concatenate results in one vector
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

function CNN.prepareDataForConvolution(dataset, convType, filterMinWidth, filterMaxWidth)
  if convType == "narrow" then
    return dataset
  elseif convType == "wide" then
    return CNN.addPadding(dataset, filterMinWidth, filterMaxWidth)
  end
end

function CNN.test()
  -- CNN hyperparameters
  local batchSize = 15
  local filterMinWidth = 1
  local filterMaxWidth = 10
  local inputSize = 17
  local alphabetLen = 25
  local charEmbeddingLen = 5
  local convolutionType = "narrow"
--  local convolutionType = "wide"

  -- Dummy data similar to our real dataset.
  local t = torch.Tensor(batchSize, inputSize)
  local dataset = {
    t:random(1, alphabetLen):clone(),
    t:random(1, alphabetLen):clone(),
    t:random(1, alphabetLen):clone(),
    t:random(1, alphabetLen):clone()
  }
  print(dataset)

  local input = CNN.prepareDataForConvolution(dataset, convolutionType, filterMinWidth, filterMaxWidth)
  print("input", input)

  local net = CNN.parallelConvolution(convolutionType, alphabetLen, charEmbeddingLen, inputSize, filterMinWidth, filterMaxWidth)
  local result = net:forward(input)
  print("result", result)

  -- checking for number of batch elements.
  print("Number of elements in batch. Test passed:", assert(result[3]:size(1) == 15))

  -- checking for output features number.
  print("Output features number. Test passed:", assert(result[4]:size(2) == 125))
end

return CNN