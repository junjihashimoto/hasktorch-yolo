{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Vision.Metrics where

import Data.List (sort, sortBy)

type Recall = Float
type Precision = Float
type Ap = Float
type F1 = Float

type Confidence = Float
type TP = Bool

computeAP :: [(Confidence,TP)] -> Int -> (Recall,Precision,F1,Ap)
computeAP predicted_bounding_box num_of_ground_truth_box =
  computeAP' $ computeRecallAndPrecision predicted_bounding_box num_of_ground_truth_box

computeAP' :: [(Recall,Precision)] -> (Recall,Precision,F1,Ap)
computeAP' [] = (0,0,0,0)
computeAP' pairs =
  let ordered = sort pairs
      (r,p) = last ordered
      pairs' = [(0,0)] ++ ordered ++ [(1,0)]
      mpre = scanr (\(r',p') (_,v)-> (r', Prelude.max v p')) (0,0) pairs'
      ap = foldl (\v ((r1,p1),(r0,_)) -> v + (r1-r0)*p1) 0 $ zip (tail mpre) mpre
  in (r,p,2*r*p/(p+r+1e-16),ap)


computeRecallAndPrecision :: [(Confidence,TP)] -> Int -> [(Recall,Precision)]
computeRecallAndPrecision predicted_bounding_box num_of_ground_truth_box =
  let comp (a0,b0) (a1,b1) =
        let i = compare a1 a0
        in if i == EQ then compare b1 b0 else i
      pairs' = sortBy comp predicted_bounding_box
      ng = fromIntegral num_of_ground_truth_box :: Float
      pairs'' = tail $ scanl (\(s,_) (i,(_,tp)) ->
                         let ns = if tp then s+1.0 else s
                         in (ns,(ns/ng, ns/i)))
                (0.0,(0.0,0.0))
                $ zip [1.0,2.0..] pairs'
  in map snd pairs''
