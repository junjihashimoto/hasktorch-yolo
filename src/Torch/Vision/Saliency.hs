module Torch.Vision.Saliency where

import Torch.Tensor
import Torch.Autograd
import qualified Torch.Functional as D
import Torch.TensorFactories
import Control.Monad (foldM)

smoothGrad :: Int -> Float -> (Tensor -> Tensor) -> Tensor -> IO Tensor
smoothGrad num_samples standard_deviation func input_image = do
  v <- foldM loop init' [1..num_samples]
  return $ (1.0 / fromIntegral num_samples :: Float) `D.mulScalar` v 
  where
    image_shape = shape input_image
    init' = zeros' image_shape
    loop sum' _ = do
      r <- randnIO' image_shape
      input <- makeIndependent $ input_image + (standard_deviation `D.mulScalar` r)
      let loss = func $ toDependent input
          (g:_) = grad loss [input]
      return $ sum' + D.abs g
