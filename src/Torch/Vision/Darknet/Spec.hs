{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE RecordWildCards #-}

module Torch.Vision.Darknet.Spec where

import Control.Monad (forM)
import Data.Map (Map)
import GHC.Exts
import Torch.NN
import qualified Torch.Vision.Darknet.Config as C

data LayerSpec
  = LConvolutionSpec ConvolutionSpec
  | LConvolutionWithBatchNormSpec ConvolutionWithBatchNormSpec
  | LMaxPoolSpec MaxPoolSpec
  | LAvgPoolSpec AvgPoolSpec
  | LSoftMaxSpec SoftMaxSpec
  | LUpSampleSpec UpSampleSpec
  | LRouteSpec RouteSpec
  | LShortCutSpec ShortCutSpec
  | LYoloSpec YoloSpec
  deriving (Show, Eq)

data ConvolutionSpec = ConvolutionSpec
  { input_filters :: Int,
    filters :: Int,
    layer_size :: Int,
    stride :: Int,
    activation :: String
  }
  deriving (Show, Eq)

data ConvolutionWithBatchNormSpec = ConvolutionWithBatchNormSpec
  { input_filters :: Int,
    filters :: Int,
    layer_size :: Int,
    stride :: Int,
    activation :: String
  }
  deriving (Show, Eq)

data AvgPoolSpec = AvgPoolSpec
  deriving (Show, Eq)

data SoftMaxSpec = SoftMaxSpec
  deriving (Show, Eq)

data MaxPoolSpec = MaxPoolSpec
  { input_filters :: Int,
    layer_size :: Int,
    stride :: Int
  }
  deriving (Show, Eq)

data RouteSpec = RouteSpec
  { input_filters :: Int,
    layers :: [Int]
  }
  deriving (Show, Eq)

data ShortCutSpec = ShortCutSpec
  { input_filters :: Int,
    from :: Int,
    activation :: String
  }
  deriving (Show, Eq)

data YoloSpec = YoloSpec
  { input_filters :: Int,
    anchors :: [(Int, Int)],
    classes :: Int,
    img_size :: Int
  }
  deriving (Show, Eq)

type Index = Int

data DarknetSpec = DarknetSpec (Map Index LayerSpec)
  deriving (Show)

toDarknetSpec :: C.DarknetConfig -> Either String DarknetSpec
toDarknetSpec (C.DarknetConfig global layer_configs) = do
  layers <- forM (toList layer_configs) $ \(idx, (layer, inputSize, outputSize)) ->
    let input_filters = inputSize
     in case layer of
          C.Convolution {..} ->
            if batch_normalize
              then pure $ (idx, (LConvolutionWithBatchNormSpec $ ConvolutionWithBatchNormSpec {..}))
              else pure $ (idx, (LConvolutionSpec $ ConvolutionSpec {..}))
          C.MaxPool {..} -> pure $ (idx, (LMaxPoolSpec $ MaxPoolSpec {..}))
          C.AvgPool -> pure $ (idx, (LAvgPoolSpec $ AvgPoolSpec))
          C.SoftMax -> pure $ (idx, (LSoftMaxSpec $ SoftMaxSpec))
          C.UpSample {..} -> pure $ (idx, (LUpSampleSpec $ UpSampleSpec {upsampleInputFilters = input_filters, upsampleStride = stride}))
          C.Route {..} ->
            pure $
              ( idx,
                ( LRouteSpec $
                    RouteSpec
                      { input_filters = input_filters,
                        layers = map (\i -> if i < 0 then idx + i else i) layers
                      }
                )
              )
          C.ShortCut {..} ->
            pure $
              ( idx,
                ( LShortCutSpec $
                    ShortCutSpec
                      { input_filters = input_filters,
                        from = if from < 0 then idx + from else from,
                        activation = activation
                      }
                )
              )
          C.Yolo {..} ->
            pure $
              ( idx,
                ( LYoloSpec $
                    YoloSpec
                      { input_filters = input_filters,
                        anchors = map (\i -> anchors !! i) mask,
                        classes = classes,
                        img_size = C.height global
                      }
                )
              )
  pure $ DarknetSpec (fromList layers)
