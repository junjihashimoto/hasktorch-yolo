{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

module TrainingSpec (spec, main) where

import Control.Exception.Safe
import Control.Monad.State.Strict
import qualified Data.ByteString.Lazy as B
import qualified Data.Map as M
import Data.Word
import GHC.Exts
import GHC.Generics
import qualified System.IO
import Test.Hspec
import Torch.Functional
import qualified Torch.Functional.Internal as I
import Torch.NN
import Torch.Serialize
import Torch.Tensor
import Torch.TensorFactories
import Torch.Typed.NN (HasForward (..))
import Torch.Vision.Darknet.Config
import Torch.Vision.Darknet.Forward
import Torch.Vision.Darknet.Spec

main = hspec spec

readFloatTensor :: FilePath -> [Int] -> IO Tensor
readFloatTensor file shape =
  System.IO.withFile file System.IO.ReadMode $ \h -> do
    loadBinary h (zeros' shape)

spec :: Spec
spec = do
  describe "Yolov3" $ do
    it "LossForTraining" $ do
      mconfig <- readIniFile "config/yolov3.cfg"
      let Right cfg@(DarknetConfig global layers) = mconfig
      length (toList layers) `shouldBe` 107
      let Right spec = toDarknetSpec cfg
      net <- sample spec
      net' <- loadWeights net "weights/yolov3.weights"
      imgs <- readFloatTensor "test-data/training/imgs.bin" [2, 3, 480, 480]
      targets <- readFloatTensor "test-data/training/targets.bin" [16, 6]
      exp_loss <- readFloatTensor "test-data/training/loss.bin" []
      exp_outputs <- readFloatTensor "test-data/training/outputs.bin" [2, 14175, 85]
      exp_layer0 <- readFloatTensor "test-data/training/layer0.bin" [2, 32, 480, 480]
      exp_layer1 <- readFloatTensor "test-data/training/layer1.bin" [2, 64, 240, 240]
      exp_layer81 <- readFloatTensor "test-data/training/layer81.bin" [2, 255, 15, 15]
      exp_layer105 <- readFloatTensor "test-data/training/layer105.bin" [2, 255, 60, 60]
      let (outputs, loss) = forwardDarknet net' (Just targets, imgs)
          layer0 = outputs M.! 0
          layer1 = outputs M.! 1
          layer81 = outputs M.! 81
          layer105 = outputs M.! 105
      asValue (mseLoss exp_layer0 layer0) < (0.000001 :: Float) `shouldBe` True
      asValue (mseLoss exp_layer1 layer1) < (0.0001 :: Float) `shouldBe` True
      asValue (mseLoss exp_layer81 layer81) < (0.0001 :: Float) `shouldBe` True
      asValue (mseLoss exp_layer105 layer105) < (0.0001 :: Float) `shouldBe` True
      asValue (mseLoss exp_loss loss) < (0.0001 :: Float) `shouldBe` True
