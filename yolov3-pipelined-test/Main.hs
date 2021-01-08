{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Codec.Picture as I
import Control.Exception.Safe
import Control.Monad (foldM, forM, forM_, when)
import qualified Data.Map as M
import Data.Maybe (catMaybes)
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

import Pipes hiding (cat)
import Pipes hiding (cat)
import Pipes.Concurrent

import Control.Concurrent.Async (async, wait)
import Control.Concurrent (threadDelay)
import Control.Monad.Writer.Lazy
import System.Environment

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


makeBatchedDatasets :: MonadIO m => Datasets -> Producer (Maybe (Int,[FilePath])) m ()
makeBatchedDatasets datasets = loop (zip [0..] (makeBatch 16 $ valid datasets))
  where
    loop [] = do
      yield Nothing
    loop (x@(i,_):xs) = do
      yield (Just x)
      loop xs

makeBatchedImages :: MonadIO m => Device -> Pipe (Maybe (Int,[FilePath])) (Maybe ([[BBox]],Tensor)) m ()
makeBatchedImages device = loop
  where
    loop = await >>= \case
      Nothing -> do
        yield Nothing
      Just (!i, !batch) -> do
        liftIO $ do
          performGC
          print (i,"readImage")
        imgs' <- liftIO $ forM batch $ \file -> do
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
            input_data = cat (Dim 0) $ map snd imgs :: Tensor
        yield $ Just (btargets, _toDevice device input_data)
        loop

doInference :: MonadIO m => Darknet -> Pipe (Maybe ([[BBox]],Tensor)) (Maybe ([[BBox]],Tensor)) m ()
doInference net' = loop
  where
    loop = do
      await >>= \case
        Nothing -> do
          yield Nothing
        Just (!btargets,!input_data) -> do
          liftIO $ print "start:inference"
          inferences <- liftIO $ do
            performGC
            detach $ toCPU $ snd (forwardDarknet net' (Nothing, input_data))
          liftIO $ print "end:inference"
          yield $ Just (btargets,inferences)
          loop

doNonMaxSuppression :: MonadIO m => Pipe (Maybe ([[BBox]],Tensor)) (Maybe ([(BBox, (Confidence, TP))],[BBox])) m ()
doNonMaxSuppression =
  await >>= \case
    Nothing -> do
      yield Nothing
    Just (!btargets, !inferences) -> do
      liftIO $ print "nonMaxSuppression"
      let boutputs = batchedNonMaxSuppression inferences 0.001 0.5
          v = flip map (zip btargets boutputs) $ \(!targets, !outputs) ->
            let inference_bbox = flip map outputs $ \output ->
                  let [!x0, !y0, !x1, !y1, !object_confidence, !class_confidence, !classid, !ids] = asValue output :: [Float]
                  in (BBox (round classid) x0 y0 x1 y1, object_confidence)
             in Just (computeTPForBBox 0.5 targets inference_bbox,targets)
      each v
      doNonMaxSuppression        

type Ret = [([(BBox, (Confidence, TP))],[BBox])]
--maybeToList :: Monad m => Consumer (Maybe a) m [a]
maybeToList :: MonadIO m => Consumer (Maybe ([(BBox, (Confidence, TP))],[BBox])) (WriterT Ret m) ()
maybeToList = loop
  where
    loop = do
      await >>= \case
        Just !x -> do
          tell [x]
          loop
        Nothing -> return ()

saveResult :: MonadIO m => Consumer (Maybe ([(BBox, (Confidence, TP))],[BBox])) m ()
saveResult = loop
  where
    loop = do
      await >>= \case
        Just !x -> do
          liftIO $ appendFile "results.txt" (show x)
          loop
        Nothing -> return ()



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
  net <- sample spec
  net' <- toDevice device <$> loadWeights net weight_file

  datasets <-
    readDatasets datasets_file >>= \case
      Right (cfg :: Datasets) -> return cfg
      Left err -> throwIO $ userError err


  v <- execWriterT $ runEffect $
    makeBatchedDatasets datasets >->
    makeBatchedImages device >->
    doInference net' >->
    doNonMaxSuppression >->
    maybeToList
{-
  (out0, in0) <- spawn $ bounded 1
  w0 <- async $ do
    runEffect $
      makeBatchedDatasets datasets >->
      toOutput out0

  (out1, in1) <- spawn $ bounded 1
  w1 <- forM [1..1] $ \i ->
    async $ do
      runEffect $
        fromInput in0  >->
        makeBatchedImages >->
        toOutput out1

  (out2, in2) <- spawn $ bounded 1
  w2 <- forM [1..1] $ \i ->
    async $ do
      runEffect $
        fromInput in1  >->
        doInference net' >->
        toOutput out2

  v <- execWriterT $ runEffect $
      fromInput in2 >->
      doNonMaxSuppression >->
      maybeToList
  mapM_ wait ([w0] ++ w1 ++ w2)
-}

  let targets = concat $ map snd v :: [BBox]
      inferences = concat $ map fst v :: [(BBox, (Confidence, TP))]
  aps <- forM (computeAPForBBox' targets inferences) $ \(cid, (_, _, _, ap)) -> do
    print (cid, ap)
    return ap
  print $ "mAP:" ++ show (Prelude.sum aps / fromIntegral (length aps))
