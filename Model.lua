require "rnn"
require "gnuplot"
require "optim"

local fun     = require "fun"
local Text    = require "Text"
local Batch   = require "Batch"
local Storage = require "Storage"
local Helpers = require "Helpers"
local CNN     = require "CNN"


local backend = 'cuda'
--local backend = 'cl'

if backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
elseif backend == 'cl' then
  require 'cltorch'
  require 'clnn'
end

function xyToGPU(x,y)

  if backend == 'cuda' then
    local xCuda = fun
    .iter(x)
    :map(function (x) return x:float():cuda() end)
    :totable()

    -- Note the x[1] here. This is due to https://github.com/torch/cutorch/issues/227.
    -- Only one target can be provided per batch element.
    -- TODO What are the consequences of only keeping the first element?
    local yCuda = fun
    .iter(y)
    :map(function (x) return x[1]:float():cuda() end)
    :totable()
    return xCuda, yCuda
  elseif backend == 'cl' then
    return x,y -- TODO: openCL
  end

  return x,y

end

function forwardBackwardPass(model, x, y, criterion)
  -- Make `x` and `y` CUDA tensors

  local xCuda, yCuda = xyToGPU(x, y)

  local prediction = model:forward(xCuda)

  -- Use criterion to compute the loss and its gradients
  local loss        = criterion:forward (prediction, yCuda)
  local gradOutputs = criterion:backward(prediction, yCuda)

  -- The recurrent layer is memorising its gradOutputs
  model:backward(xCuda, gradOutputs)

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

function createModel(inputSize, hiddenSize, outputSize, filterMinWidth, filterMaxWidth)
  local cnnModule = CNN.getParallelConvolution(inputSize, filterMinWidth, filterMaxWidth)
  local lstmInputSize = torch.range(filterMinWidth, filterMaxWidth):sum()

  local rnnModule = nn.Sequential()
  rnnModule:add(nn.Linear(lstmInputSize, hiddenSize))
  rnnModule:add(nn.LSTM(hiddenSize, hiddenSize))
  rnnModule:add(nn.Dropout(0.5))
  rnnModule:add(nn.LSTM(hiddenSize, hiddenSize))
  rnnModule:add(nn.Dropout(0.5))
  rnnModule:add(nn.Linear(hiddenSize, outputSize))
  rnnModule:add(nn.LogSoftMax())  -- For ClassNLLCriterion
  local rnnModuleDecorated = nn.Sequencer(rnnModule)

  local model = nn.Sequential()
  model:add(cnnModule)
  model:add(rnnModuleDecorated)

  if backend == 'cuda' then
    return model:cuda()
  elseif backend == 'cl' then
    return model:cl()
  end

  return model
end

function createCriterion()
  local baseCriterion = nn.ClassNLLCriterion() --:cuda() -- FIXME: should this additional call to cuda be here?
  local criterion     = nn.SequencerCriterion(baseCriterion)
  if backend == 'cuda' then
    return criterion:cuda()
  elseif backend == 'cl' then
    return criterion:cl()
  end
  return criterion
end

function train(model, batch, epochs, learningRate, updateParameters)
  local criterion     = createCriterion()

  -- For each epoch iterate over the entire sequence
  fun.range(1, epochs):each(function (epoch)
    print("# Epoch " .. epoch .. "/" .. epochs)
    local x, y = batch:getBatches(DataSet.TrainingSet)

    fun.range(1, #x):each(function (curBatch)
      print("## Batch " .. curBatch)

      -- Obtain array of 2D tensors
      local sequencesX = Helpers.tensorToArray(x[curBatch])
      local sequencesY = Helpers.tensorToArray(y[curBatch])

      local prediction, loss =
        updateParameters(model, sequencesX, sequencesY, criterion, learningRate)

      print("Loss: ", loss)
    end)
  end)
end

function dumpBatches(batch, nBatches, nSequences)
  local x, y = batch:getBatches(DataSet.TrainingSet)

  fun.range(1, nBatches):each(function (batchNumber)
    print("# Batch: " .. batchNumber)

    fun.range(1, nSequences):each(function (sequence)
      local sequenceX = x[batchNumber][sequence]
      local sequenceY = y[batchNumber][sequence]

      print("Sequence " .. sequence)
      print("x = " .. batch:sequenceToText(sequenceX))
      print("")
      print("y = " .. batch:sequenceToText(sequenceY))
      print("")
    end)
  end)
end

if not Storage.filesExist("data") then
  Text.preprocess("input.txt")
end

local batchSize      = 20  -- Number of sequences in a batch, trained in parallel
local sequenceLength = 15  -- Number of time steps of each sequence
local epochs         = 50
local hiddenSize     = 512
local learningRate   = 0.05

-- CNN hyperparameters
local filterMinWidth = 1
local filterMaxWidth = 10

-- local updateFunction = updateParametersSGD
local updateFunction = updateParametersDefault

local batch = Batch("data", batchSize, sequenceLength)

-- dumpBatches(batch, 5, 5)

local inputSize  = batch.maximumTokenLength
local outputSize = #batch.symbols  -- Equivalent to number of classes
local model      = createModel(inputSize, hiddenSize, outputSize, filterMinWidth, filterMaxWidth)

print("Model:")
print(model)

train(model, batch, epochs, learningRate, updateFunction)
