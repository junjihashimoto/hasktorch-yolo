{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Torch.Vision.Metrics where

import Control.Exception.Safe (tryIO)
import Control.Monad (forM)
import Data.List (maximumBy, sort, sortBy)
import Data.List.Split
import qualified Data.Set as S
import GHC.Generics
import Control.DeepSeq

type Recall = Float

type Precision = Float

type Ap = Float

type F1 = Float

type Confidence = Float

type TP = Bool

type ClassID = Int

data BoundingBox = BoundingBox
  { bboxClassid :: ClassID,
    bboxX :: Float,
    bboxY :: Float,
    bboxWidth :: Float,
    bboxHeight :: Float
  }
  deriving (Show, Eq)

data BBox = BBox
  { classid :: ClassID,
    x0 :: Float,
    y0 :: Float,
    x1 :: Float,
    y1 :: Float
  }
  deriving (Show, Eq, Generic, NFData)

readBoundingBox :: FilePath -> IO [BoundingBox]
readBoundingBox file = do
  mstr <- tryIO $ readFile file
  case mstr of
    Right str -> do
      let dats = map (splitOn " ") (lines str)
      forM dats $ \(classid' : x' : y' : w' : h' : _) -> do
        let cid = read classid'
            x = read x'
            y = read y'
            w = read w'
            h = read h'
        return $ BoundingBox cid x y w h
    Left err -> return []

padding :: Int -> Int -> Int -> Int -> (Int, Int)
padding iw ih w h =
  let t0w = iw * h `div` ih
   in if (t0w > w)
        then (0, (h - (ih * w `div` iw)) `div` 2)
        else ((w - (iw * h `div` ih)) `div` 2, 0)

rescale :: Int -> Int -> Int -> Int -> BoundingBox -> BoundingBox
rescale iw ih w h (BoundingBox classid x' y' w' h') =
  let (pw, ph) = padding iw ih w h
      h_ = fromIntegral h
      w_ = fromIntegral w
      ph_ = fromIntegral ph
      pw_ = fromIntegral pw
      ny = (ph_ + y' * (h_ - ph_ * 2)) / h_
      nh = (h' * (h_ - ph_ * 2)) / h_
      nx = (pw_ + x' * (w_ - pw_ * 2)) / w_
      nw = (w' * (w_ - pw_ * 2)) / w_
   in BoundingBox classid nx ny nw nh

toXYXY :: Int -> Int -> BoundingBox -> BBox
toXYXY width height (BoundingBox classid x' y' w' h') =
  let x = fromIntegral width * x' :: Float
      y = fromIntegral height * y' :: Float
      w = fromIntegral width * w' :: Float
      h = fromIntegral height * h' :: Float
      x0 = x - w / 2
      y0 = y - h / 2
      x1 = x + w / 2
      y1 = y + h / 2
   in BBox classid x0 y0 x1 y1

bboxIou ::
  BBox ->
  BBox ->
  Float
bboxIou (BBox _ b0_x0 b0_y0 b0_x1 b0_y1) (BBox _ b1_x0 b1_y0 b1_x1 b1_y1) =
  let inter_rect_x0 = max b0_x0 b1_x0
      inter_rect_y0 = max b0_y0 b1_y0
      inter_rect_x1 = min b0_x1 b1_x1
      inter_rect_y1 = min b0_y1 b1_y1
      clampMin v = if v < 0 then 0 else v
      inter_area = clampMin (inter_rect_x1 - inter_rect_x0 + 1) * clampMin (inter_rect_y1 - inter_rect_y0 + 1)
      b0_area = (b0_x1 - b0_x0 + 1) * (b0_y1 - b0_y0 + 1)
      b1_area = (b1_x1 - b1_x0 + 1) * (b1_y1 - b1_y0 + 1)
      sum' = 1e-16 + (b0_area + b1_area - inter_area)
   in inter_area / sum'

computeTPForBBox :: Float -> [BBox] -> [(BBox, Confidence)] -> [(BBox, (Confidence, TP))]
computeTPForBBox _ [] _ = []
computeTPForBBox iou_thresh targets inference_bbox = loop S.empty inference_bbox
  where
    targets' = zip [0 ..] targets :: [(Int, BBox)]
    classes = S.fromList $ map classid targets
    maximumBy' func = maximumBy (\a b -> compare (func a) (func b))
    loop _ [] = []
    loop used_targets ((bbox, conf) : other) =
      let (iou', (id', _)) = maximumBy' fst $ map (\a@(_, t) -> (bboxIou bbox t, a)) targets'
       in if not (S.member (classid bbox) classes)
            then (bbox, (conf, False)) : loop used_targets other
            else
              if iou' < iou_thresh
                then (bbox, (conf, False)) : loop used_targets other
                else
                  if S.member id' used_targets
                    then (bbox, (conf, False)) : loop used_targets other
                    else (bbox, (conf, True)) : loop (S.insert id' used_targets) other

computeAPForBBox' :: [BBox] -> [(BBox, (Confidence, TP))] -> [(ClassID, (Recall, Precision, F1, Ap))]
computeAPForBBox' [] _ = []
computeAPForBBox' targets tp = map func classes
  where
    classes = S.toList $ S.fromList $ map classid targets
    func cls =
      let n_gt = length $ filter (\v -> classid v == cls) targets
          tp' = filter (\(v, _) -> classid v == cls) tp
       in (cls, computeAP (map snd tp') n_gt)

computeAPForBBox :: Float -> [BBox] -> [(BBox, Confidence)] -> [(ClassID, (Recall, Precision, F1, Ap))]
computeAPForBBox _ [] _ = []
computeAPForBBox iou_thresh targets inference_bbox = map func classes
  where
    classes = S.toList $ S.fromList $ map classid targets
    tp = computeTPForBBox iou_thresh targets inference_bbox
    func cls =
      let n_gt = length $ filter (\v -> classid v == cls) targets
          tp' = filter (\(v, _) -> classid v == cls) tp
       in (cls, computeAP (map snd tp') n_gt)

computeAP :: [(Confidence, TP)] -> Int -> (Recall, Precision, F1, Ap)
computeAP predicted_bounding_box num_of_ground_truth_box =
  computeAP' $ computeRecallAndPrecision predicted_bounding_box num_of_ground_truth_box

computeAP' :: [(Recall, Precision)] -> (Recall, Precision, F1, Ap)
computeAP' [] = (0, 0, 0, 0)
computeAP' pairs =
  let ordered = sort pairs
      (r, p) = last ordered
      pairs' = [(0, 0)] ++ ordered ++ [(1, 0)]
      mpre = scanr (\(r', p') (_, v) -> (r', Prelude.max v p')) (0, 0) pairs'
      ap = foldl (\v ((r1, p1), (r0, _)) -> v + (r1 - r0) * p1) 0 $ zip (tail mpre) mpre
   in (r, p, 2 * r * p / (p + r + 1e-16), ap)

computeRecallAndPrecision :: [(Confidence, TP)] -> Int -> [(Recall, Precision)]
computeRecallAndPrecision predicted_bounding_box num_of_ground_truth_box =
  let comp (a0, b0) (a1, b1) =
        let i = compare a1 a0
         in if i == EQ then compare b1 b0 else i
      pairs' = sortBy comp predicted_bounding_box
      ng = fromIntegral num_of_ground_truth_box :: Float
      pairs'' =
        tail $
          scanl
            ( \(s, _) (i, (_, tp)) ->
                let ns = if tp then s + 1.0 else s
                 in (ns, (ns / ng, ns / i))
            )
            (0.0, (0.0, 0.0))
            $ zip [1.0, 2.0 ..] pairs'
   in map snd pairs''
