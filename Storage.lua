local Storage = {}

function Storage.vocabularyFile(dataDir)
  return path.join(dataDir, 'vocabulary.t7')
end

function Storage.tensorFile(dataDir)
  return path.join(dataDir, 'data.t7')
end

function Storage.filesExist(dataDir)
  return path.exists(Storage.vocabularyFile(dataDir)) and
      path.exists(Storage.tensorFile(dataDir))
end

return Storage
