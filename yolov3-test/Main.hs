{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Main where

import qualified Codec.Picture as I
import Control.DeepSeq
import Control.Exception.Safe
import Control.Monad (foldM, forM, forM_, when)
import qualified Data.Map as M
import Data.Maybe (catMaybes)
import System.Environment (getArgs, getEnv)
import System.Mem (performGC)
import Torch hiding (conv2d, indexPut)
import qualified Torch.Functional.Internal as I
import Torch.Serialize
import Torch.Vision
import Torch.Vision.Darknet.Config
import Torch.Vision.Darknet.Forward
import Torch.Vision.Darknet.Spec
import Torch.Vision.Datasets
import Torch.Vision.Metrics
import Foreign.Ptr

foreign import ccall unsafe "malloc.h malloc_stats"
  malloc_stats  :: IO ()

foreign import ccall unsafe "malloc.h malloc_trim"
  malloc_trim  :: Int -> IO ()

--foreign import ccall unsafe "jemalloc/jemalloc.h malloc_stats_print"
--jemalloc_stats  :: Ptr () -> Ptr () -> Ptr () -> IO ()

labels :: [String]
labels =
  [ "person",
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

pmap :: Parameterized f => f -> (Parameter -> Parameter) -> f
pmap model func = replaceParameters model (map func (flattenParameters model))

pmapM :: Parameterized f => f -> (Parameter -> IO Parameter) -> IO f
pmapM model func = do
  params <- mapM func (flattenParameters model)
  return $ replaceParameters model params

toEval :: Parameterized f => f -> IO f
toEval model = pmapM model $ \p -> makeIndependentWithRequiresGrad (toDependent p) False

makeBatch :: Int -> [a] -> [[a]]
makeBatch num_batch datasets =
  let (a, ax) = (Prelude.take num_batch datasets, Prelude.drop num_batch datasets)
   in if length a < num_batch
        then a : []
        else a : makeBatch num_batch ax

readImage :: FilePath -> Int -> Int -> IO (Either String (Int, Int, Tensor))
readImage file width height =
  I.readImage file >>= \case
    Left err -> return $ Left err
    Right img' -> do
      let rgb8 = I.convertRGB8 img'
          img = (resizeRGB8 width height True) rgb8
      return $ Right (I.imageWidth rgb8, I.imageHeight rgb8, fromDynImage . I.ImageRGB8 $ img)

main = do
  args <- getArgs
  when (length args /= 3) $ do
    putStrLn "Usage: yolov3-test config-file weight-file datasets-file"
  let config_file = args !! 0
      weight_file = args !! 1
      datasets_file = args !! 2
  deviceStr <- try (getEnv "DEVICE") :: IO (Either SomeException String)
  let device = case deviceStr of
        Right "cpu" -> Device CPU 0
        Right "cuda:0" -> Device CUDA 0
        Right device -> error $ "Unknown device setting: " ++ device
        _ -> Device CPU 0

  spec <-
    readIniFile config_file >>= \case
      Right cfg@(DarknetConfig global layers) -> do
        case toDarknetSpec cfg of
          Right spec -> return spec
          Left err -> throwIO $ userError err
      Left err -> throwIO $ userError err
  net' <- do
    net <- sample spec
    net_ <- toDevice device <$> loadWeights net weight_file
    toEval net_

  datasets <-
    readDatasets datasets_file >>= \case
      Right (cfg) -> return cfg
      Left err -> throwIO $ userError err
  v <- forM (zip [0 ..] (makeBatch 16 $ valid datasets)) $ \(i, batch) -> do
    imgs' <- forM batch $ \file -> do
      bboxes <- readBoundingBox $ toLabelPath file
      Main.readImage file 416 416 >>= \case
        Right (width, height, input_tensor) -> do
          return $
            Just
              ( map (toXYXY 416 416 . rescale width height 416 416) bboxes,
                divScalar (255 :: Float) $ toType Float $ hwc2chw input_tensor
              )
        Left err -> return Nothing
    let imgs = catMaybes imgs'
        btargets = map fst imgs :: [[BBox]]
        input_data = _toDevice device $ cat (Dim 0) $ map snd imgs :: Tensor
    inferences <- detach $ toCPU $ snd $ forwardDarknet net' (Nothing, input_data)
    performGC
    malloc_trim 0
    print $ (i, shape inferences)
    let boutputs = batchedNonMaxSuppression inferences 0.001 0.5
    forM (zip btargets boutputs) $ \(!targets, !outputs) -> do
      !inference_bbox <- forM (zip [0 ..] outputs) $ \(i, output) -> do
        let [!x0, !y0, !x1, !y1, !object_confidence, !class_confidence, !classid, !ids] = asValue output :: [Float]
        return $ (BBox (round classid) x0 y0 x1 y1, object_confidence)
      let !comp = (computeTPForBBox 0.5 targets inference_bbox, targets)
      return comp
  let targets = concat $ map snd (concat v) :: [BBox]
      inferences = concat $ map fst (concat v) :: [(BBox, (Confidence, TP))]
  aps <- forM (computeAPForBBox' targets inferences) $ \(cid, (_, _, _, ap)) -> do
    print (cid, ap)
    return ap
  print $ "mAP:" ++ show (Prelude.sum aps / fromIntegral (length aps))
