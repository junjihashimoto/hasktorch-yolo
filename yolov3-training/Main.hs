{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Codec.Picture as I
import Control.Concurrent (threadDelay)
import Control.Concurrent.Async (async, wait)
import Control.Exception.Safe
import Control.Monad (foldM, forM, forM_, when)
import Control.Monad.Writer.Lazy
import qualified Data.Map as M
import Data.Maybe (catMaybes)
import Pipes hiding (cat)
import Pipes.Concurrent
import System.Environment
import System.Environment (getArgs)
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
import Torch.Optim.CppOptim
-- import Torch.Optim
import Data.Default.Class

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

makeBatchedDatasets :: MonadIO m => Datasets -> Producer (Maybe (Int, [FilePath])) m ()
makeBatchedDatasets datasets = loop (zip [0 ..] (makeBatch 16 $ train datasets))
  where
    loop [] = do
      yield Nothing
    loop (x@(i, _) : xs) = do
      yield (Just x)
      loop xs

makeBatchedImages :: MonadIO m => Device -> Pipe (Maybe (Int, [FilePath])) (Maybe (Tensor, Tensor)) m ()
makeBatchedImages device = loop
  where
    loop =
      await >>= \case
        Nothing -> do
          yield Nothing
        Just (!i, !batch) -> do
          liftIO $ do
            performGC
            print (i, "readImage")
          imgs' <- liftIO $
            forM batch $ \file -> do
              bboxes <- readBoundingBox $ toLabelPath file
              Main.readImage file 416 416 >>= \case
                Right (width, height, input_tensor) -> do
                  return $
                    Just
                      ( map (rescale width height 416 416) bboxes,
                        divScalar (255 :: Float) $ toType Float $ hwc2chw input_tensor
                      )
                Left err -> return Nothing
          let imgs = catMaybes imgs'
              btargets = map fst imgs :: [[BoundingBox]]
              input_data = cat (Dim 0) $ map snd imgs :: Tensor
          yield $ Just (contiguous $ boundingbox2Tensor btargets, contiguous $ _toDevice device input_data)
          loop

training :: (MonadIO m, Optimizer opt) => String -> String -> opt -> Darknet -> Pipe (Maybe (Tensor, Tensor)) (Maybe Float) m ()
training weight_file saved_weight_file optState org_net = loop (org_net,optState)
  where
    loop (net,opt) = do
      await >>= \case
        Nothing -> do
          liftIO $ print "Save the weight file for the trained model"
          liftIO $ saveWeights net weight_file saved_weight_file
          yield Nothing
        Just (!btargets, !input_data) -> do
          liftIO $ print "start:training"
          let loss = snd (forwardDarknet net (Just btargets, input_data))
          (net',opt') <- liftIO $ runStep net opt loss 5e-4
          liftIO $ print "end:training"
          yield $ Just (asValue loss)
          loop (net',opt')

maybeToList :: MonadIO m => Consumer (Maybe a) (WriterT [a] m) ()
maybeToList = loop
  where
    loop = do
      await >>= \case
        Just !x -> do
          tell [x]
          loop
        Nothing -> return ()

pmap :: Parameterized f => f -> (Parameter -> Parameter) -> f
pmap model func = replaceParameters model (map func (flattenParameters model))

pmapM :: Parameterized f => f -> (Parameter -> IO Parameter) -> IO f
pmapM model func = do
  params <- mapM func (flattenParameters model)
  return $ replaceParameters model params

toEval :: Parameterized f => f -> IO f
toEval model = pmapM model $ \p -> makeIndependentWithRequiresGrad (toDependent p) False

toTrain :: Parameterized f => f -> IO f
toTrain model = pmapM model $ \p -> makeIndependent (toDependent p)

main = do
  args <- getArgs
  when (length args /= 3) $ do
    putStrLn "Usage: yolov3-training config-file weight-file datasets-file"
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
  net <- sample spec
  net' <- toDevice device <$> loadWeights net weight_file

  datasets <-
    readDatasets datasets_file >>= \case
      Right (cfg :: Datasets) -> return cfg
      Left err -> throwIO $ userError err

  let numEpoch = 100
      adamOpt = (def { adamLr = 1e-4
                     , adamBetas = (0.9, 0.999)
                     , adamEps = 1e-8
                     , adamWeightDecay = 0
                     , adamAmsgrad = False
                     } :: AdamOptions)

  optimizer <- initOptimizer adamOpt net'
--   let optimizer = GD

  forM_ [1..numEpoch] $ \epoch -> do
    v <-
      execWriterT $
        runEffect $
          makeBatchedDatasets datasets
            >-> makeBatchedImages device
            >-> training weight_file "saved_weight.weights" optimizer net'
            >-> maybeToList
    print (epoch,v)
