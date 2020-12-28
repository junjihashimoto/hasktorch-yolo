{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE ExtendedDefaultRules #-}

module Main where

import qualified Codec.Picture as I
import Control.Monad (forM_, when, foldM)
import Control.Exception.Safe
import Torch hiding (conv2d, indexPut)
import Torch.Vision
import Torch.Vision.Darknet.Config
import Torch.Vision.Darknet.Forward
import Torch.Vision.Darknet.Spec
import System.Environment (getArgs)
import qualified Data.Map as M

labels :: [String]
labels = [
  "person",
  "bicycle",
  "car",
  "motorbike",
  "aeroplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "sofa",
  "pottedplant",
  "bed",
  "diningtable",
  "toilet",
  "tvmonitor",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush"
  ]

main = do
  args <- getArgs
  when (length args /= 4) $ do
    putStrLn "Usage: yolov3 config-file weight-file input-image-file output-image-file"
  let config_file = args !! 0
      weight_file = args !! 1
      input_file = args !! 2
      output_file = args !! 3
      device = Device CUDA 0
      toDev = _toDevice device
      toHost = _toDevice (Device CPU 0)
    
  mconfig <- readIniFile config_file
  spec <- case mconfig of
    Right cfg@(DarknetConfig global layers) -> do
      case toDarknetSpec cfg of
        Right spec -> return spec
        Left err -> throwIO $ userError err
    Left err -> throwIO $ userError err
  net <- sample spec
  net' <- loadWeights net weight_file
  
  readImageAsRGB8WithScaling input_file 416 416 True >>= \case
    Right (input_image, input_tensor) -> do
      let input_data' = divScalar (255 :: Float) (hwc2chw $ toType Float input_tensor)
          (outs,out) = forwardDarknet net' (Nothing, input_data')
          outputs = nonMaxSuppression out 0.8 0.4
      forM_ (zip [0..] outputs) $ \(i, output) -> do
        let [x0,y0,x1,y1,object_confidence,class_confidence,classid,ids] = map truncate (asValue output :: [Float])
        drawString (show i ++ " " ++ labels !! classid) (x0+1) (y0+1) (255,255,255) (0,0,0) input_image
        drawRect x0 y0 x1 y1 (255,255,255) input_image
      I.writePng output_file input_image
    Left err -> print err
