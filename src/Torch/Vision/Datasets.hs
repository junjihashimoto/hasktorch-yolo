{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Vision.Datasets where

import Data.Ini.Config
import qualified Data.Text as T
import qualified Data.Text.IO as T

data Datasets' = Datasets'
  { classes' :: Int
  , train' :: FilePath
  , valid' :: FilePath
  , names' :: FilePath
  } deriving (Show, Eq)

data Datasets = Datasets
  { classes :: Int
  , train :: [FilePath]
  , valid :: [FilePath]
  , names :: [String]
  } deriving (Show, Eq)

configParser :: IniParser Datasets'
configParser =
  section "datasets" $
    Datasets'
      <$> fieldOf "classes" number
      <*> fieldOf "train" string
      <*> fieldOf "valid" string
      <*> fieldOf "names" string

readDatasets :: String -> IO (Either String Datasets)
readDatasets filepath = do
  contents <- T.readFile filepath
  case parseIniFile contents configParser of
    Left err -> return $ Left err
    Right (Datasets' {..}) -> do
      r <- Datasets
           <$> pure classes'
           <*> (lines <$> Prelude.readFile train')
           <*> (lines <$> Prelude.readFile valid')
           <*> (lines <$> Prelude.readFile names')
      return $ Right r
  
toLabelPath :: FilePath -> FilePath
toLabelPath file =
  T.unpack $
  T.replace "images" "labels" $
  T.replace ".jpg" ".txt" $ 
  T.replace ".png" ".txt" $
  T.pack file
