--
-- Obtain tensor representation of text
--

require "lfs"
local fun     = require "fun"
local rex     = require "rex_pcre"
local utf8    = require "lua-utf8"
local Storage = require "Storage"
local Helpers = require "Helpers"

Symbols = {
  WordStart = '{',
  WordEnd   = '}',
  ZeroPad   = ' '
}

DataSet = {
  TrainingSet   = 1,
  TestSet       = 2,
  ValidationSet = 3
}

local Text = {}

function splitSentences(text)
  return fun
    .iter(Helpers.toArray(rex.gmatch(text .. ".", "[^:;\\.?!]+[:;\\.?!]+")))
    :map(Helpers.trim)
end

function splitWords(sentence)
  return fun.iter(Helpers.toArray(rex.split(sentence, "\\s+|/")))
end

-- Cut off words at 25 characters. Prepend `WordStart` and append `WordEnd`,
-- delete all occurrences of these two tokens in `word`.
function Text.tokeniseWord(word)
  if #word > 25 then
    word = word:sub(1, 25)
  end

  return fun.chain(
    Symbols.WordStart,
    fun.iter(Helpers.stringToArray(word))
      :map(utf8.char)
      :filter(function (cur)
        local filter = {
          [Symbols.WordStart] = 1,
          [Symbols.WordEnd]   = 1,
          [Symbols.ZeroPad]   = 1
        }
        return filter[cur] == nil
      end),
    Symbols.WordEnd)
end

-- Creates a character lookup table from tokenised words
function Text.lookupTable(words)
  -- List of collected characters
  local characters = { Symbols.ZeroPad, Symbols.WordStart, Symbols.WordEnd }

  -- Maps characters to indexes in chracters
  local characterToIndex = {
    [Symbols.ZeroPad]    = 1,
    [Symbols.WordStart]  = 2,
    [Symbols.WordEnd]    = 3
  }

  words:each(function (word)
    word:each(function (character)
      if characterToIndex[character] == nil then
        characters[#characters + 1] = character
        characterToIndex[character] = #characters
      end
    end)
  end)

  return characters, characterToIndex
end

function Text.vectoriseTokens(tokens, indexes)
  local maximumTokenLength = tokens:map(function (x) return x:length() end):max()

  -- All values are one and thus refer to Symbols.ZeroPad
  local charsTensor = torch.ones(tokens:length(), maximumTokenLength)

  -- TODO zipWithIndex is missing in luafun
  tokens:zip(fun.range(tokens:length())):each(function (token, tokenId)
    token:zip(fun.range(token:length())):each(function (character, charId)
      local characterIndex = indexes[character]
      charsTensor[tokenId][charId] = characterIndex
    end)
  end)

  return charsTensor
end

function Text.wordsTensor(data)
  local tokens = Helpers.flatMap(splitSentences(data), splitWords)
    :map(Text.tokeniseWord)
  local characters, characterToIndex = Text.lookupTable(tokens)
  local tensor = Text.vectoriseTokens(tokens, characterToIndex)

  -- print("Tokens: "               .. tensor:size(1))
  -- print("Maximum token length: " .. tensor:size(2))

  return characters, characterToIndex, tensor
end

function Text.charsTensor(string)
  local characters       = {}
  local characterToIndex = {}

  local table = fun.iter(string):map(function (character)
    if characterToIndex[character] == nil then
      characters[#characters + 1] = character
      characterToIndex[character] = #characters
    end

    return characterToIndex[character]
  end):totable()

  local tensor = torch.Tensor(table)

  return characters, characterToIndex, tensor
end

function saveFiles(dataDir, characters, characterToIndex, tensor)
  local vocabularyFile = Storage.vocabularyFile(dataDir)
  local tensorFile     = Storage.tensorFile(dataDir)

  print('Saving ' .. vocabularyFile)
  torch.save(vocabularyFile, { characters, characterToIndex })

  print('Saving ' .. tensorFile)
  torch.save(tensorFile, tensor)
end

function Text.splitTensor(tensor)
  -- TODO Is there a better way in Torch to split the tensor?
  local trainingSize   = math.floor(tensor:size(1) * 0.50)
  local testSize       = math.floor(tensor:size(1) * 0.25)
  local validationSize = math.floor(tensor:size(1) * 0.25)

  local trainingSet = tensor:sub(1, trainingSize)

  local testSet = tensor:sub(
    trainingSize + 1,
    trainingSize + 1 + testSize)

  local validationSet = tensor:sub(
    trainingSize + testSize + 1,
    math.min(tensor:size(1), trainingSize + testSize + 1 + validationSize))

  return trainingSet, testSet, validationSet
end

function Text.preprocess(fileName, tensorFunction, length)
  local char, charToIndex, tensor = tensorFunction(
    Helpers.readFile(fileName):sub(1, length))

  local training, test, validation = Text.splitTensor(tensor)

  local tensors = {
    [DataSet.TrainingSet]   = training,
    [DataSet.TestSet]       = test,
    [DataSet.ValidationSet] = validation
  }

  lfs.mkdir("data")
  saveFiles("data/", char, charToIndex, tensors)
end

return Text
