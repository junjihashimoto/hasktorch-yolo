{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}

module MetricsSpec (spec, main) where

import Control.Exception.Safe
import Test.Hspec
import Torch
import Torch.Vision.Metrics

main = hspec spec

-- See https://github.com/rafaelpadilla/Object-Detection-Metrics
recall_vs_precision :: [(Float,Float)]
recall_vs_precision = [
    (0.0666 , 1       ),
    (0.0666 , 0.5     ),
    (0.1333 , 0.6666  ),
    (0.1333 , 0.5     ),
    (0.1333 , 0.4     ),
    (0.1333 , 0.3333  ),
    (0.1333 , 0.2857  ),
    (0.1333 , 0.25    ),
    (0.1333 , 0.2222  ),
    (0.2    , 0.3     ),
    (0.2    , 0.2727  ),
    (0.2666 , 0.3333  ),
    (0.3333 , 0.3846  ),
    (0.4    , 0.4285  ),
    (0.4    , 0.4     ),
    (0.4    , 0.375   ),
    (0.4    , 0.3529  ),
    (0.4    , 0.3333  ),
    (0.4    , 0.3157  ),
    (0.4    , 0.3     ),
    (0.4    , 0.2857  ),
    (0.4    , 0.2727  ),
    (0.4666 , 0.3043  ),
    (0.4666 , 0.2916  )
    ]

confidence_tp = [
  (0.88, False),
  (0.70, True),
  (0.80, False),
  (0.71, False),
  (0.54, True),
  (0.74, False),
  (0.18, True),
  (0.67, False),
  (0.38, False),
  (0.91, True),
  (0.44, False),
  (0.35, False),
  (0.78, False),
  (0.45, False),
  (0.14, False),
  (0.62, True),
  (0.44, False),
  (0.95, True),
  (0.23, False),
  (0.45, False),
  (0.84, False),
  (0.43, False),
  (0.48, True),
  (0.95, False)
  ]

spec :: Spec
spec = do
  describe "Metrics for computer vision" $ do
    it "AP" $ do
      let (_,_,_,ap) = computeAP' recall_vs_precision
      ap `shouldBe` 0.24560957
    it "AP" $ do
      computeRecallAndPrecision confidence_tp 15 `shouldBe`
        [ (6.666667e-2,1.0),
          (6.666667e-2,0.5),
          (0.13333334,0.6666667),
          (0.13333334,0.5),
          (0.13333334,0.4),
          (0.13333334,0.33333334),
          (0.13333334,0.2857143),
          (0.13333334,0.25),
          (0.13333334,0.22222222),
          (0.2,0.3),(0.2,0.27272728),
          (0.26666668,0.33333334),
          (0.33333334,0.3846154),
          (0.4,0.42857143),
          (0.4,0.4),
          (0.4,0.375),
          (0.4,0.3529412),
          (0.4,0.33333334),
          (0.4,0.31578946),
          (0.4,0.3),
          (0.4,0.2857143),
          (0.4,0.27272728),
          (0.46666667,0.3043478),
          (0.46666667,0.29166666)
        ]
