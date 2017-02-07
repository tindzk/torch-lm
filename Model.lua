require "rnn"
require "gnuplot"
require "optim"

local fun     = require "fun"
local Text    = require "Text"
local Batch   = require "Batch"
local Storage = require "Storage"
local Helpers = require "Helpers"
local CNN     = require "CNN"
local Highway = require "Highway"

-- local backend = "cuda"
-- local backend = "cl"
local backend = "cpu"

if backend == "cuda" then
  require "cunn"
elseif backend == "cl" then
  require "clnn"
end

function xyToGPU(x, y)
  local xGpu = fun
    .iter(x)
    :map(function (x)
      if backend == "cuda" then
        return x:float():cuda()
      elseif backend == "cl" then
        return x:float():cl()
      else
        return x
      end
    end)
    :totable()

  local yGpu = fun
    .iter(y)
    :map(function (x)
      if backend == "cuda" then
        return x:float():cuda()
      elseif backend == "cl" then
        return x:float():cl()
      else
        return x
      end
    end)
    :totable()

  return xGpu, yGpu
end

function forwardBackwardPass(model, x, y, criterion)
  -- Make `x` and `y` CUDA/OpenCL tensors
  local xGpu, yGpu = xyToGPU(x, y)
  local prediction = model:forward(xGpu)

  -- Use criterion to compute the loss and its gradients
  local loss        = criterion:forward (prediction, yGpu)
  local gradOutputs = criterion:backward(prediction, yGpu)

  -- The recurrent layer is memorising its gradOutputs
  model:backward(xGpu, gradOutputs)

  return prediction, loss
end

function updateParametersDefault(model, input, target, criterion, learningRate)
  local prediction, loss = forwardBackwardPass(model, input, target, criterion)

  model:updateParameters(learningRate)
  model:zeroGradParameters()

  return prediction, loss
end

local sgdState = {}
function updateParametersSGD(model, input, target, criterion, learningRate)
  -- Obtain weights and gradients from model
  local modelParams, modelGradParams = model:getParameters()

  local sgdParams = {
    learningRate      = learningRate,
    learningRateDecay = 1e-4,
    weightDecay       = 0,
    momentum          = 0
  }

  local prediction = {}

  -- Compute value of the loss function at `input` and its gradient
  local eval = function(newModelParams)
    -- Set the new weights in the model if they have changed
    if modelParams ~= newModelParams then modelParams:copy(newModelParams) end

    -- Reset gradients; gradients are always accumulated to accommodate batch
    -- methods.
    modelGradParams:zero()

    local _prediction, loss =
      forwardBackwardPass(model, input, target, criterion)
    prediction[1] = _prediction

    return loss, modelGradParams
  end

  local _, fs = optim.sgd(eval, modelParams, sgdParams, sgdState)
  local loss = fs[1]

  return prediction[1][1], loss
end

function rnnModule(inputSize, hiddenSize, outputSize, dropout)
  local rnnModule = nn.Sequential()
  rnnModule:add(nn.Linear(inputSize, hiddenSize))
  rnnModule:add(nn.LSTM(hiddenSize, hiddenSize))
  rnnModule:add(nn.Dropout(dropout))
  rnnModule:add(nn.LSTM(hiddenSize, hiddenSize))
  rnnModule:add(nn.Dropout(dropout))
  rnnModule:add(nn.Linear(hiddenSize, outputSize))
  rnnModule:add(nn.LogSoftMax())  -- For ClassNLLCriterion
  return rnnModule
end

function createModel(convolutionType, alphabetLen, charEmbeddingLen, inputSize,
                     hiddenSize, outputSize, filterMinWidth, filterMaxWidth,
                     highwayLayers, dropout)
  local lstmInputSize = torch.range(
    inputSize - (filterMaxWidth - filterMinWidth), inputSize):sum()

  local cnnModule = nil
  if convolutionType == "narrow" then
    cnnModule = CNN.narrowConvolution(inputSize, alphabetLen, charEmbeddingLen,
       filterMinWidth, filterMaxWidth, backend)
  end

  local highwayModule = Highway.mlp(lstmInputSize, highwayLayers)
  highwayModule.name = "highway"

  local model = nn.Sequencer(
    nn.Sequential()
      :add(cnnModule)
      :add(highwayModule)
      :add(rnnModule(lstmInputSize, hiddenSize, outputSize, dropout)))

  if backend == "cuda" then
    return model:cuda()
  elseif backend == "cl" then
    return model:cl()
  else
    return model
  end
end

function createCriterion()
  local baseCriterion = nn.ClassNLLCriterion()
  local criterion     = nn.SequencerCriterion(baseCriterion)

  if backend == "cuda" then
    return criterion:cuda()
  elseif backend == "cl" then
    return criterion:cl()
  end

  return criterion
end

function train(model, convolutionType, batch, epochs, learningRate, updateParameters, filterMinWidth, filterMaxWidth)
  local criterion = createCriterion()

  -- For each epoch iterate over the entire sequence
  fun.range(1, epochs):each(function (epoch)
    print("# Epoch " .. epoch .. "/" .. epochs)
    local x, y = batch:getBatches(DataSet.TrainingSet)

    fun.range(1, #x):each(function (curBatch)
      print("## Batch " .. curBatch)

      -- Obtain array of 2D tensors
      local sequencesX  = Helpers.tensorToArray(x[curBatch])
      local inputX      = sequencesX
      if convolutionType == "wide" then
        inputX = CNN.addPadding(sequencesX, filterMinWidth, filterMaxWidth)
      end

      local sequencesY  = Helpers.tensorToArray(y[curBatch])

      local prediction, loss = updateParameters(model, inputX, sequencesY, criterion, learningRate)

      print("Loss: ", loss)
    end)
  end)
end

function dumpBatches(batch, nBatches, nSequences)
  local x, y = batch:getBatches(DataSet.TrainingSet)

  fun.range(1, nBatches):each(function (batchNumber)
    print("# Batch: " .. batchNumber)

    fun.range(1, nSequences):each(function (sequence)
      print("Sequence " .. sequence)
      local sequenceX = x[batchNumber][sequence]
      local sequenceY = y[batchNumber][sequence]

      print("x = " .. batch:toText(sequenceX))
      print("")
      print("y = " .. batch:toText(sequenceY))
      print("")
    end)
  end)
end

if not Storage.filesExist("data") then
  Text.preprocess("input.txt", Text.charsTensor, 1000000)
end

local batchSize      = 20   -- Number of sequences in a batch, trained in parallel
local sequenceLength = 100  -- Number of time steps of each sequence
local epochs         = 50
local hiddenSize     = 512
local learningRate   = 0.05
local dropout        = 0.5

-- CNN hyperparameters
local convolutionType  = "narrow"
local filterMinWidth   = 1
local filterMaxWidth   = 10
local charEmbeddingLen = 5

-- Highway hyperparameters
local highwayLayers = 1

-- local updateFunction = updateParametersSGD
local updateFunction = updateParametersDefault

local batch = Batch("data", batchSize, sequenceLength)

-- dumpBatches(batch, 5, 5)

local inputSize   = sequenceLength
local outputSize  = #batch.symbols  -- Equivalent to number of classes
local alphabetLen = #batch.symbols
local model       = createModel(convolutionType, alphabetLen, charEmbeddingLen,
                                inputSize, hiddenSize, outputSize,
                                filterMinWidth, filterMaxWidth, highwayLayers,
                                dropout)

print("Model:")
print(model)

train(model, convolutionType, batch, epochs, learningRate, updateFunction, filterMinWidth, filterMaxWidth)
