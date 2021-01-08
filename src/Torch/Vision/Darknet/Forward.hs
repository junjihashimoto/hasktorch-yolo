{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

module Torch.Vision.Darknet.Forward where

import Control.Monad (forM, join)
import qualified Data.ByteString as BS
import Data.Map (Map, empty, insert)
import qualified Data.Map as M
import Data.Maybe (catMaybes, isJust)
import GHC.Exts
import GHC.Generics
import qualified System.IO
import Torch.Autograd
import Torch.DType as D
import qualified Torch.Functional as D
import qualified Torch.Functional.Internal as I
import Torch.NN
import Torch.Serialize
import Torch.Tensor as D
import Torch.TensorFactories
--import Torch.Typed.NN (HasForward (..))
import qualified Torch.Vision.Darknet.Spec as S

type Index = Int

type Loss = Tensor

data ConvolutionWithBatchNorm = ConvolutionWithBatchNorm
  { conv2d :: Conv2d,
    batchNorm :: BatchNorm,
    stride :: Int,
    layerSize :: Int,
    isLeaky :: Bool
  }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.ConvolutionWithBatchNormSpec ConvolutionWithBatchNorm where
  sample S.ConvolutionWithBatchNormSpec {..} = do
    ConvolutionWithBatchNorm
      <$> sample
        ( Conv2dSpec
            { inputChannelSize = input_filters,
              outputChannelSize = filters,
              kernelHeight = layer_size,
              kernelWidth = layer_size
            }
        )
      <*> sample
        ( BatchNormSpec
            { numFeatures = filters
            }
        )
      <*> pure stride
      <*> pure layer_size
      <*> pure (activation == "leaky")

instance HasForward ConvolutionWithBatchNorm (Bool, Tensor) Tensor where
  forward ConvolutionWithBatchNorm {..} (train, input) =
    let pad = (layerSize - 1) `div` 2
        activation = if isLeaky then flip I.leaky_relu 0.1 else id
     in activation $
          batchNormForward batchNorm train 0.90000000000000002 1.0000000000000001e-05 $
            conv2dForward conv2d (stride, stride) (pad, pad) input
  forwardStoch f a = pure $ forward f a

data Convolution = Convolution
  { conv2d :: Conv2d,
    stride :: Int,
    layerSize :: Int,
    isLeaky :: Bool
  }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.ConvolutionSpec Convolution where
  sample S.ConvolutionSpec {..} = do
    Convolution
      <$> sample
        ( Conv2dSpec
            { inputChannelSize = input_filters,
              outputChannelSize = filters,
              kernelHeight = layer_size,
              kernelWidth = layer_size
            }
        )
      <*> pure stride
      <*> pure layer_size
      <*> pure (activation == "leaky")

instance HasForward Convolution Tensor Tensor where
  forward Convolution {..} input =
    let pad = (layerSize - 1) `div` 2
     in conv2dForward conv2d (stride, stride) (pad, pad) input
  forwardStoch f a = pure $ forward f a

data MaxPool = MaxPool
  { stride :: Int,
    layerSize :: Int
  }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.MaxPoolSpec MaxPool where
  sample S.MaxPoolSpec {..} = do
    MaxPool
      <$> pure stride
      <*> pure layer_size

instance HasForward MaxPool Tensor Tensor where
  forward MaxPool {..} input =
    let pad = (layerSize - 1) `div` 2
        out = D.maxPool2d (layerSize, layerSize) (stride, stride) (pad, pad) (1, 1) D.Floor input
        [n, f, h, w] = shape out
        padx = D.cat (D.Dim (-1)) [out, zeros' [n, f, h, 1]]
        pady = D.cat (D.Dim (2)) [padx, zeros' [n, f, 1, w + 1]]
        out' = if layerSize == 2 && stride == 1 then pady else out
     in out'
  forwardStoch f a = pure $ forward f a

data AvgPool = AvgPool
  deriving (Show, Generic, Parameterized)

instance Randomizable S.AvgPoolSpec AvgPool where
  sample S.AvgPoolSpec = pure AvgPool

instance HasForward AvgPool Tensor Tensor where
  forward AvgPool input =
    let out = D.adaptiveAvgPool2d (1, 1) input
     in out
  forwardStoch f a = pure $ forward f a

data SoftMax = SoftMax
  deriving (Show, Generic, Parameterized)

instance Randomizable S.SoftMaxSpec SoftMax where
  sample S.SoftMaxSpec = pure SoftMax

instance HasForward SoftMax Tensor Tensor where
  forward SoftMax input =
    let out = D.softmax (D.Dim 1) input
     in out
  forwardStoch f a = pure $ forward f a

data Route = Route
  { layers :: [Int]
  }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.RouteSpec Route where
  sample S.RouteSpec {..} = do
    Route
      <$> pure layers

instance HasForward Route (Map Int Tensor) Tensor where
  forward Route {..} inputs =
    D.cat (D.Dim 1) (map (inputs M.!) layers)
  forwardStoch f a = pure $ forward f a

data ShortCut = ShortCut
  { from :: Int,
    isLeaky :: Bool
--    shortCutWeight :: Tensor,
--    shortCutBias :: Tensor
  }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.ShortCutSpec ShortCut where
  sample S.ShortCutSpec {..} = do
    ShortCut
      <$> pure from
      <*> pure (activation == "leaky")

instance HasForward ShortCut (Tensor, Map Int Tensor) Tensor where
  forward ShortCut {..} (input, inputs) =
    let [_, c0, x0, y0] = D.shape input
        shortcut = inputs M.! from
        [_, c1, x1, y1] = D.shape shortcut
        activation = if isLeaky then flip I.leaky_relu 0.1 else id
    in if c0 == c1 && x0 == x1 && y0 == y1
       then activation $ input + shortcut
       else
         let stride = if x1 < x0 then 1 else x1 `div` x0
             zero2d :: Int -> [[Float]]
             zero2d n = replicate n $ replicate n 0
             one2d :: Int -> [[Float]]
             one2d n = (1 : replicate (n -1) 0) : (replicate (n -1) $ replicate n 0)
             weight :: [[[[Float]]]]
             weight = do
               o <- [0 .. (c0 -1)]
               return $ do
                 i <- [0 .. (c1 -1)]
                 if o == i
                   then return $ one2d stride
                   else return $ zero2d stride
             dev = device input
             weight' = _toDevice dev $ asTensor weight
          in activation $ input + D.conv2d' weight' (_toDevice dev $ zeros' [c0]) (stride, stride) (0, 0) shortcut
  forwardStoch f a = pure $ forward f a

type Anchors = [(Float, Float)]

type ScaledAnchors = [(Float, Float)]

data Yolo = Yolo
  { anchors :: Anchors,
    classes :: Int,
    img_size :: Int
  }
  deriving (Show, Generic, Parameterized)

instance Randomizable S.YoloSpec Yolo where
  sample S.YoloSpec {..} =
    pure $
      Yolo
        { classes = classes,
          anchors = map (\(a, b) -> (fromIntegral a, fromIntegral b)) anchors,
          img_size = img_size
        }

newtype Prediction = Prediction {fromPrediction :: Tensor} deriving (Show)

toPrediction :: Yolo -> Tensor -> Prediction
toPrediction Yolo {..} input =
  let num_samples = D.size 0 input
      grid_size = D.size 2 input
      num_anchors = length anchors
   in Prediction $ D.contiguous $ D.permute [0, 1, 3, 4, 2] $ D.reshape [num_samples, num_anchors, classes + 5, grid_size, grid_size] input

squeezeLastDim :: Tensor -> Tensor
squeezeLastDim input = I.squeezeDim input (-1)

toX ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toX prediction = D.sigmoid $ squeezeLastDim (D.slice (-1) 0 1 1 $ fromPrediction prediction)

toY ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toY prediction = D.sigmoid $ squeezeLastDim (D.slice (-1) 1 2 1 $ fromPrediction prediction)

toW ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toW prediction = squeezeLastDim (D.slice (-1) 2 3 1 $ fromPrediction prediction)

toH ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toH prediction = squeezeLastDim (D.slice (-1) 3 4 1 $ fromPrediction prediction)

toPredConf ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid]
  Tensor
toPredConf prediction = D.sigmoid $ squeezeLastDim (D.slice (-1) 4 5 1 $ fromPrediction prediction)

toPredClass ::
  -- | [batch, anchors, grid, grid, class + 5]
  Prediction ->
  -- |  [batch, anchors, grid, grid, class]
  Tensor
toPredClass prediction =
  let input = fromPrediction prediction
      num_features = D.size (-1) input
   in D.sigmoid $ squeezeLastDim (D.slice (-1) 5 num_features 1 input)

gridX ::
  -- |  grid size
  Int ->
  -- |  [1, 1, grid, grid]
  Tensor
gridX g = D.reshape [1, 1, g, g] $ D.repeat [g, 1] $ arange' (0 :: Int) g (1 :: Int)

gridY ::
  -- |  grid size
  Int ->
  -- |  [1, 1, grid, grid]
  Tensor
gridY g = D.contiguous $ D.reshape [1, 1, g, g] $ I.t $ D.repeat [g, 1] $ arange' (0 :: Int) g (1 :: Int)

toScaledAnchors :: Anchors -> Float -> ScaledAnchors
toScaledAnchors anchors stride = map (\(a_w, a_h) -> (a_w / stride, a_h / stride)) anchors

toAnchorW :: ScaledAnchors -> Tensor
toAnchorW scaled_anchors = D.reshape [1, length scaled_anchors, 1, 1] $ asTensor $ (map fst scaled_anchors :: [Float])

toAnchorH :: ScaledAnchors -> Tensor
toAnchorH scaled_anchors = D.reshape [1, length scaled_anchors, 1, 1] $ asTensor $ (map snd scaled_anchors :: [Float])

toPredBox ::
  Yolo ->
  Prediction ->
  Float ->
  (Tensor, Tensor, Tensor, Tensor)
toPredBox Yolo {..} prediction stride =
  let input = fromPrediction prediction
      dev = device input
      grid_size = D.size 2 input
      scaled_anchors = toScaledAnchors anchors stride
      anchor_w = _toDevice dev $ toAnchorW scaled_anchors
      anchor_h = _toDevice dev $ toAnchorH scaled_anchors
   in ( toX prediction + _toDevice dev (gridX grid_size),
        toY prediction + _toDevice dev (gridY grid_size),
        D.exp (toW prediction) * anchor_w,
        D.exp (toH prediction) * anchor_h
      )

bboxWhIou ::
  (Float, Float) ->
  -- | (batch, batch)
  (Tensor, Tensor) ->
  -- | batch
  Tensor
bboxWhIou (w1', h1') (w2, h2) =
  let w1 = asTensor w1'
      h1 = asTensor h1'
      inter_area = I.min w1 w2 * I.min h1 h2
      union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
   in inter_area / union_area

bboxIou ::
  Tensor ->
  Tensor ->
  Tensor
bboxIou box1 box2 =
  let b1_x1 = box1 ! (Ellipsis, 0)
      b1_y1 = box1 ! (Ellipsis, 1)
      b1_x2 = box1 ! (Ellipsis, 2)
      b1_y2 = box1 ! (Ellipsis, 3)
      b2_x1 = box2 ! (Ellipsis, 0)
      b2_y1 = box2 ! (Ellipsis, 1)
      b2_x2 = box2 ! (Ellipsis, 2)
      b2_y2 = box2 ! (Ellipsis, 3)
      inter_rect_x1 = I.max b1_x1 b2_x1
      inter_rect_y1 = I.max b1_y1 b2_y1
      inter_rect_x2 = I.min b1_x2 b2_x2
      inter_rect_y2 = I.min b1_y2 b2_y2
      inter_area = D.clampMin 0 (inter_rect_x2 - inter_rect_x1 + 1) * D.clampMin 0 (inter_rect_y2 - inter_rect_y1 + 1)
      b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
      b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
   in inter_area / (b1_area + b2_area - inter_area + 1e-16)

data Target = Target
  { obj_mask :: Tensor,
    noobj_mask :: Tensor,
    tx :: Tensor,
    ty :: Tensor,
    tw :: Tensor,
    th :: Tensor,
    tcls :: Tensor,
    tconf :: Tensor
  }

toBuildTargets ::
  (Tensor, Tensor, Tensor, Tensor) ->
  Tensor ->
  Tensor ->
  Anchors ->
  Float ->
  Target
toBuildTargets
  ( pred_boxes_x,
    pred_boxes_y,
    pred_boxes_w,
    pred_boxes_h
    )
  pred_cls
  target
  anchors
  ignore_thres =
    let nB = D.size 0 pred_boxes_x
        nA = D.size 1 pred_boxes_x
        nC = D.size (-1) pred_cls
        nG = D.size 2 pred_boxes_x
        obj_mask_init = zeros [nB, nA, nG, nG] bool_opts
        noobj_mask_init = ones [nB, nA, nG, nG] bool_opts
        class_mask_init = zeros' [nB, nA, nG, nG]
        iou_scores_init = zeros' [nB, nA, nG, nG]
        tx_init = zeros' [nB, nA, nG, nG]
        ty_init = zeros' [nB, nA, nG, nG]
        tcls_init = zeros' [nB, nA, nG, nG, nC]
        target_boxes = nG `D.mulScalar` (D.slice (-1) 2 6 1 target)
        gx = squeezeLastDim $ D.slice (-1) 0 1 1 target_boxes
        gy = squeezeLastDim $ D.slice (-1) 1 2 1 target_boxes
        gw = squeezeLastDim $ D.slice (-1) 2 3 1 target_boxes
        gh = squeezeLastDim $ D.slice (-1) 3 4 1 target_boxes
        gi = toType D.Int64 gx
        gj = toType D.Int64 gy
        -- (anchors,batch)
        ious_list = map (\anchor -> bboxWhIou anchor (gw, gh)) anchors
        ious = D.transpose (D.Dim 0) (D.Dim 1) $ D.stack (D.Dim 0) ious_list
        (best_ious, best_n) = I.maxDim ious 0 False
        best_n_anchor = anchors !! (asValue best_n :: Int)
        b = toType D.Int64 $ squeezeLastDim $ D.slice (-1) 0 1 1 target
        target_labels = toType D.Int64 $ squeezeLastDim $ D.slice (-1) 1 2 1 target
        obj_mask =
          maskedFill
            obj_mask_init
            (b, best_n, gj, gi)
            True
        noobj_mask' =
          maskedFill
            noobj_mask_init
            (b, best_n, gj, gi)
            False
        noobj_mask =
          foldl
            ( \v i ->
                maskedFill
                  v
                  (b ! i, (ious `D.gt` (asTensor ignore_thres)) ! 0, gj ! i, gi ! i)
                  False
            )
            noobj_mask'
            [0 .. ((D.size 0 gj) -1)]
        tx =
          maskedFill
            (zeros' [nB, nA, nG, nG])
            (b, best_n, gj, gi)
            (gx - D.floor gx)
        ty =
          maskedFill
            (zeros' [nB, nA, nG, nG])
            (b, best_n, gj, gi)
            (gy - D.floor gy)
        tw =
          maskedFill
            (zeros' [nB, nA, nG, nG])
            (b, best_n, gj, gi)
            (I.log (gw / (asTensor (fst best_n_anchor)) + 1e-16))
        th =
          maskedFill
            (zeros' [nB, nA, nG, nG])
            (b, best_n, gj, gi)
            (I.log (gh / (asTensor (snd best_n_anchor)) + 1e-16))
        tcls =
          maskedFill
            tcls_init
            (b, best_n, gj, gi, target_labels)
            (1 :: Float)
        tconf = toType D.Float obj_mask
     in Target {..}

index :: TensorLike a => Tensor -> [a] -> Tensor
index org idx = I.index org (map asTensor idx)

indexPut :: (TensorLike a, TensorLike b) => Tensor -> [a] -> b -> Tensor
indexPut org idx value = I.index_put org (map asTensor idx) (asTensor value) False

totalLoss :: Yolo -> Prediction -> Target -> Tensor
totalLoss yolo prediction Target {..} =
  let x = toX prediction
      y = toY prediction
      w = toW prediction
      h = toH prediction
      pred_conf = toPredConf prediction
      pred_cls = toPredClass prediction
      omask t = t `index` [obj_mask]
      nmask t = t `index` [noobj_mask]
      loss_x = D.mseLoss (omask x) (omask ty)
      loss_y = D.mseLoss (omask y) (omask ty)
      loss_w = D.mseLoss (omask w) (omask tw)
      loss_h = D.mseLoss (omask h) (omask th)
      bceLoss = D.binaryCrossEntropyLoss'
      loss_conf_obj = bceLoss (omask pred_conf) (omask tconf)
      loss_conf_noobj = bceLoss (nmask pred_conf) (nmask tconf)
      obj_scale = 1
      noobj_scale = 100
      loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj
      loss_cls = bceLoss (omask pred_cls) (omask tcls)
   in loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

data YoloOutput = YoloOutput
  { x :: Tensor,
    y :: Tensor,
    w :: Tensor,
    h :: Tensor,
    predBoxes :: Tensor,
    predConf :: Tensor,
    predClass :: Tensor
  }
  deriving (Show)

instance HasForward Yolo (Maybe Tensor, Tensor) Tensor where
  forwardStoch f a = pure $ forward f a
  forward yolo@Yolo {..} (train, input) =
    let num_samples = D.size 0 input
        grid_size = D.size 2 input
        stride = (fromIntegral img_size) / (fromIntegral grid_size) :: Float
        prediction = toPrediction yolo input
        pred_boxes = toPredBox yolo prediction stride
        (px, py, pw, ph) = pred_boxes
        pred_cls = toPredClass prediction
        pred_conf = toPredConf prediction
        cc = D.stack (D.Dim (-1)) [px, py, pw, ph]
     in case train of
          Nothing ->
            D.cat
              (D.Dim (-1))
              [ stride `D.mulScalar` D.view [num_samples, -1, 4] cc,
                D.view [num_samples, -1, 1] pred_conf,
                D.view [num_samples, -1, classes] pred_cls
              ]
          Just target ->
            let ignore_thres = 0.5
                build_target = toBuildTargets pred_boxes pred_cls target anchors ignore_thres
             in totalLoss yolo prediction build_target

data Layer
  = LConvolution Convolution
  | LConvolutionWithBatchNorm ConvolutionWithBatchNorm
  | LMaxPool MaxPool
  | LAvgPool AvgPool
  | LSoftMax SoftMax
  | LUpSample UpSample
  | LRoute Route
  | LShortCut ShortCut
  | LYolo Yolo
  deriving (Show, Generic, Parameterized)

data Darknet = Darknet [(Index, Layer)] deriving (Show, Generic, Parameterized)

loadWeights :: Darknet -> String -> IO Darknet
loadWeights (Darknet layers) weights_file = do
  System.IO.withFile weights_file System.IO.ReadMode $ \handle -> do
    _ <- BS.hGet handle (5 * 4) -- skip header
    layers' <- forM layers $ \(i, layer) -> do
      case layer of
        LConvolution (Convolution (Conv2d weight bias) a b c) -> do
          new_params_b <- loadBinary handle (toDependent bias) >>= makeIndependent
          new_params_w <- loadBinary handle (toDependent weight) >>= makeIndependent
          return $ (i, LConvolution (Convolution (Conv2d new_params_w new_params_b) a b c))
        LConvolutionWithBatchNorm (ConvolutionWithBatchNorm (Conv2d weight _) (BatchNorm bw bb rm rv) a b c) -> do
          new_bb <- join $ makeIndependent <$> loadBinary handle (toDependent bw)
          new_bw <- join $ makeIndependent <$> loadBinary handle (toDependent bb)
          new_rm <- toDependent <$> join (makeIndependentWithRequiresGrad <$> loadBinary handle rm <*> pure False)
          new_rv <- toDependent <$> join (makeIndependentWithRequiresGrad <$> loadBinary handle rv <*> pure False)
          let [features, _, _, _] = shape $ toDependent weight
          new_b <- makeIndependentWithRequiresGrad (zeros' [features]) False
          new_w <- join $ makeIndependent <$> loadBinary handle (toDependent weight)
          return $ (i, LConvolutionWithBatchNorm (ConvolutionWithBatchNorm (Conv2d new_w new_b) (BatchNorm new_bw new_bb new_rm new_rv) a b c))
        _ -> do
          let cur_params = flattenParameters layer
          new_params <- forM cur_params $ \param -> loadBinary handle (toDependent param) >>= makeIndependent
          return $ (i, replaceParameters layer new_params)
    return $ Darknet layers'

instance Randomizable S.DarknetSpec Darknet where
  sample (S.DarknetSpec layers) = do
    layers' <- forM (toList layers) $ \(idx, layer) ->
      case layer of
        S.LConvolutionSpec s -> (\s -> (idx, (LConvolution s))) <$> sample s
        S.LConvolutionWithBatchNormSpec s -> (\s -> (idx, (LConvolutionWithBatchNorm s))) <$> sample s
        S.LMaxPoolSpec s -> (\s -> (idx, (LMaxPool s))) <$> sample s
        S.LAvgPoolSpec s -> (\s -> (idx, (LAvgPool s))) <$> sample s
        S.LSoftMaxSpec s -> (\s -> (idx, (LSoftMax s))) <$> sample s
        S.LUpSampleSpec s -> (\s -> (idx, (LUpSample s))) <$> sample s
        S.LRouteSpec s -> (\s -> (idx, (LRoute s))) <$> sample s
        S.LShortCutSpec s -> (\s -> (idx, (LShortCut s))) <$> sample s
        S.LYoloSpec s -> (\s -> (idx, (LYolo s))) <$> sample s
    pure $ Darknet (fromList layers')

forwardDarknet :: Darknet -> (Maybe Tensor, Tensor) -> ((Map Index Tensor), Tensor)
forwardDarknet = forwardDarknet' (-1)

forwardDarknet' :: Int -> Darknet -> (Maybe Tensor, Tensor) -> ((Map Index Tensor), Tensor)
forwardDarknet' depth (Darknet layers) (train, input) = loop depth layers empty []
  where
    loop :: Int -> [(Index, Layer)] -> (Map Index Tensor) -> [Tensor] -> ((Map Index Tensor), Tensor)
    loop 0 _ maps tensors = (maps, D.cat (D.Dim 1) (reverse tensors))
    loop _ [] !maps !tensors = (maps, D.cat (D.Dim 1) (reverse tensors))
    loop !n !((idx, layer) : next) !layerOutputs !yoloOutputs =
      let input' = (if idx == 0 then input else layerOutputs M.! (idx -1))
       in case layer of
            LConvolution s ->
              let out = forward s input'
               in loop (n -1) next (insert idx out layerOutputs) yoloOutputs
            LConvolutionWithBatchNorm s ->
              let out = forward s (isJust train, input')
               in loop (n -1) next (insert idx out layerOutputs) yoloOutputs
            LMaxPool s ->
              let out = forward s input'
               in loop (n -1) next (insert idx out layerOutputs) yoloOutputs
            LAvgPool s ->
              let out = forward s input'
               in loop (n -1) next (insert idx out layerOutputs) yoloOutputs
            LSoftMax s ->
              let out = forward s input'
               in loop (n -1) next (insert idx out layerOutputs) (out : yoloOutputs)
            LUpSample s ->
              let out = forward s input'
               in loop (n -1) next (insert idx out layerOutputs) yoloOutputs
            LRoute s ->
              let out = forward s layerOutputs
               in loop (n -1) next (insert idx out layerOutputs) yoloOutputs
            LShortCut s ->
              let out = forward s (input', layerOutputs)
               in loop (n -1) next (insert idx out layerOutputs) yoloOutputs
            LYolo s ->
              let out = forward s (train, input')
               in loop (n -1) next (insert idx out layerOutputs) (out : yoloOutputs)

instance HasForward Darknet (Maybe Tensor, Tensor) Tensor where
  forward net input = snd $ forwardDarknet net input
  forwardStoch f a = pure $ forward f a

-- | Returns xyxy format from xywh
xywh2xyxy ::
  -- | input (batch,grid,4)
  Tensor ->
  -- | output (batch,grid,4)
  Tensor
xywh2xyxy xywh =
  let x = xywh ! (Ellipsis, 0)
      y = xywh ! (Ellipsis, 1)
      w = xywh ! (Ellipsis, 2)
      h = xywh ! (Ellipsis, 3)
      other = xywh ! (Ellipsis, Slice (4, None))
   in D.cat
        (D.Dim (-1))
        [ D.stack
            (D.Dim (-1))
            [ (x - (0.5 `D.mulScalar` w)),
              (y - (0.5 `D.mulScalar` h)),
              (x + (0.5 `D.mulScalar` w)),
              (y + (0.5 `D.mulScalar` h))
            ],
          other
        ]

-- | Returns objects that exceed a threshold.
toDetection ::
  -- | input
  Tensor ->
  -- | confidence threshold
  Float ->
  -- | (the number of objects that exceed the threshold)
  Maybe Tensor
toDetection prediction conf_thres =
  if head (shape mprediction') == 0
  then Nothing
  else
    let (values, indices) = D.maxDim (D.Dim (-1)) D.RemoveDim mprediction'
        list_of_detections =
          [ prediction' ! (Ellipsis, Slice (0, 5)),
            D.stack
              (D.Dim (-1))
              [ values,
                indices,
                ids
              ]
          ]
        detections = D.cat (D.Dim (-1)) list_of_detections
        score = prediction' ! (Ellipsis, 4) * values
        detections' = detections ! (I.argsort score (-1) True)
     in Just detections'
  where
    indexes = ((prediction ! (Ellipsis, 4)) `D.ge` asTensor conf_thres)
    prediction' = xywh2xyxy $ prediction ! indexes
    n = (reverse $ D.shape prediction) !! 1
    ids = (D.reshape [1, n] $ (arange' (0 :: Int) n (1 :: Int))) ! indexes
    offset_class = 5
    mprediction' = prediction' ! (Ellipsis, Slice (offset_class, None))

nonMaxSuppression ::
  Tensor ->
  Float ->
  Float ->
  [Tensor]
nonMaxSuppression prediction conf_thres nms_thres =
  case morg_detections of
    Just org_detecgtions -> loop org_detecgtions
    Nothing -> []
  where
    morg_detections = toDetection prediction conf_thres
    loop :: Tensor -> [Tensor]
    loop detections =
      if (D.size 0 detections == 0)
        then []
        else
          let detection0 = detections ! 0
              large_overlap =
                bboxIou
                  (D.unsqueeze (D.Dim 0) (detection0 ! Slice (None, 4)))
                  (detections ! (Ellipsis, Slice (None, 4)))
                  `D.ge` asTensor nms_thres
              label_match = detection0 ! 6 `D.eq` detections ! (Ellipsis, 6)
              invalid = large_overlap `I.logical_and` label_match
              weights = D.unsqueeze (D.Dim 1) $ detections ! (invalid, 4)
              detections' = detections ! (I.logical_not invalid)
              detection' = D.sumDim (D.Dim 0) D.RemoveDim (dtype prediction) (weights * (detections ! (invalid, Slice (0, 4)))) / D.sumAll weights
           in D.cat (D.Dim (-1)) [detection', detection0 ! Slice (4, None)] : loop detections'

batchedNonMaxSuppression ::
  Tensor ->
  Float ->
  Float ->
  [[Tensor]]
batchedNonMaxSuppression predictions conf_thres nms_thres = map (\p -> nonMaxSuppression p conf_thres nms_thres) $ I.split predictions 1 0

updateDarknet :: Darknet -> (Int -> Layer -> IO Layer) -> IO Darknet
updateDarknet (Darknet darknet) func = do
  v <- forM darknet $ \(i, layer) -> do
    vv <- func i layer
    return (i, vv)
  return $ Darknet v

getTensorFromDarknet :: Darknet -> (Int -> Layer -> IO (Maybe Tensor)) -> IO [Tensor]
getTensorFromDarknet (Darknet darknet) func = do
  v <- forM darknet $ \(i, layer) -> func i layer
  return $ catMaybes v

