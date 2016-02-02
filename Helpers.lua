local fun     = require "fun"
local utf8    = require "lua-utf8"
local Helpers = {}

-- Return a new tensor with the first `n` rows
function Helpers.take(tensor, n)
  local result = tensor:clone()
  return result:sub(1, n)
end

-- Slice the entire tensor by `n` rows
function Helpers.slide(tensor, n)
  local result = tensor:clone()
  return result
    :sub(1, -(n + 1))
    :copy(result:sub(n + 1, -1))
end

-- Return a tensor omitting the last row
function Helpers.init(tensor)
  return Helpers.take(tensor, tensor:size(1) - 1)
end

-- Return a tensor omitting the first row
function Helpers.tail(tensor)
  return Helpers.take(tensor, tensor:size(1) - 1)
end

-- From http://lua-users.org/wiki/StringTrim
function Helpers.trim(s)
  local from = s:match"^%s*()"
  return from > #s and "" or s:match(".*%S", from)
end

function Helpers.toArray(...)
  local arr = {}
  for v in ... do arr[#arr + 1] = v end
  return arr
end

-- Given a 3D tensor, returns an array of 2D tensors
function Helpers.tensorToArray(x)
  local result = {}
  for i = 1, x:size(1) do result[#result + 1] = x[i] end
  return result
end

-- Returns an array with each UTF-8 character as integer value
function Helpers.stringToArray(string)
  local arr = {}
  for _, char in utf8.next, string do arr[#arr + 1] = char end
  return arr
end

-- There is no flatMap() in luafun
function Helpers.flatMap(iterator, f)
  --return iterator:foldl(
  --  function (acc, cur) return acc:chain(f(cur)) end, fun.iter({}))

  local flattened = {}

  iterator:each(function (x)
    f(x):each(function (y)
      flattened[#flattened + 1] = y
    end)
  end)

  return fun.iter(flattened)
end

function Helpers.readFile(file)
  local f = io.open(file, "rb")
  local content = f:read("*all")
  f:close()
  return content
end

return Helpers
