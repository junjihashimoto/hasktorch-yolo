{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

module ImageSpec (spec, main) where

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
import Torch.Vision
import Torch.DType
import Torch.Typed.NN (HasForward (..))
import Torch.Vision.Darknet.Config
import Torch.Vision.Darknet.Forward
import Torch.Vision.Darknet.Spec

main = hspec spec

spec :: Spec
spec = do
  describe "ImageSpec" $ do
    it "load" $ do
      input_data <- System.IO.withFile "test-data/metrics/input-images.bin" System.IO.ReadMode $ \h -> do
        loadBinary h (zeros' [1, 3, 416, 416])
      Right (_, raw) <- readImageAsRGB8WithScaling "test-data/metrics/COCO_val2014_000000000164.jpg" 416 416 True
      let target = divScalar (255 :: Float) (hwc2chw $ toType Float raw)
      asValue (mseLoss input_data target) < (0.00001 :: Float) `shouldBe` True
