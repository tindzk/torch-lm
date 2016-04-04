require 'busted.runner'()

local fun  = require "fun"
local Text = require "../Text"

describe("Text", function()
  it("Vectorise tokens", function()
  local tokens = fun.iter({"a", "b", "ab"}):map(Text.tokeniseWord)
    local characters, characterToIndex = Text.lookupTable(tokens)
    assert.are.same(characters, {
      Symbols.ZeroPad,
      Symbols.WordStart,
      Symbols.WordEnd,
      "a",
      "b"
    })

  local tensor = Text.vectoriseTokens(tokens, characterToIndex)
  assert.are.same(tensor:totable(), {
    { 2, 4, 3, 1 },
    { 2, 5, 3, 1 },
    { 2, 4, 5, 3 }
  })
  end)
end)
