"""
Slope Failure Risk Calculator
=============================

Calculates risk scores for highway slope segments based on:
- InSAR deformation data
- Slope angle from DEM
- Rainfall data
- Geology

Risk Score: 0-100
Levels: GREEN (0-25), YELLOW (26-50), ORANGE (51-75), RED (76-100)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GEOLOGY_NAMES_JA = {
    'granite': '花崗岩（安定）',
    'andesite': '安山岩（安定）',
    'basalt': '玄武岩（安定）',
    'sandstone': '砂岩（やや注意）',
    'shale': '頁岩（要注意）',
    'mudstone': '泥岩（要注意）',
    'tuff': '凝灰岩（要注意）',
    'volcanic_ash': '火山灰（要注意）',
    'colluvium': '崖錐堆積物（要注意）',
    'alluvial': '沖積層（普通）',
    'fill': '盛土（要注意）',
    'landslide_deposit': '地すべり堆積物（危険）',
}


class RiskLevel(Enum):
    """Risk level classification"""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    UNKNOWN = "unknown"


@dataclass
class SegmentData:
    """
    Input data for a single highway segment.

    All measurements should be for the same segment/pixel.
    """
    segment_id: str
    lat: float
    lon: float

    # InSAR measurements
    deformation_rate_mm_year: float  # Negative = subsidence
    deformation_acceleration: float = 0.0  # mm/year^2
    coherence: float = 1.0  # InSAR quality (0-1)

    # Terrain
    slope_angle_degrees: float = 0.0
    aspect_degrees: float = 0.0  # Slope facing direction

    # Weather
    rainfall_48h_mm: float = 0.0
    rainfall_7d_mm: float = 0.0

    # Geology
    geology_type: str = "unknown"


@dataclass
class RiskResult:
    """Risk calculation result for a segment"""
    segment_id: str
    lat: float
    lon: float
    score: int
    level: RiskLevel
    component_scores: Dict[str, float]
    timestamp: datetime
    coherence: float
    message: str = ""
    is_reliable: bool = True
    insights: List[Dict] = None  # Data-driven insights for display


class RiskScoreCalculator:
    """
    Calculate slope failure risk scores.

    The algorithm combines multiple factors with configurable weights:
    - Deformation rate (35%): Primary indicator from InSAR
    - Deformation acceleration (20%): Is movement speeding up?
    - Slope angle (15%): Steeper = higher risk
    - Rainfall (15%): Recent precipitation increases risk
    - Geology (15%): Different rock/soil types have different stability

    Example:
        >>> calculator = RiskScoreCalculator()
        >>> segment = SegmentData(
        ...     segment_id="CHUO_001",
        ...     lat=35.7,
        ...     lon=138.6,
        ...     deformation_rate_mm_year=-15.0,
        ...     slope_angle_degrees=35,
        ...     rainfall_48h_mm=80
        ... )
        >>> result = calculator.calculate(segment)
        >>> print(f"Risk: {result.score} ({result.level.value})")
    """

    # Default weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        'deformation_rate': 0.35,
        'deformation_acceleration': 0.20,
        'slope_angle': 0.15,
        'rainfall_factor': 0.15,
        'geology_factor': 0.15,
    }

    # Geology risk scores (0-100)
    GEOLOGY_RISK = {
        'granite': 15,
        'andesite': 20,
        'basalt': 20,
        'sandstone': 35,
        'shale': 45,
        'mudstone': 55,
        'tuff': 50,
        'volcanic_ash': 65,
        'colluvium': 75,
        'alluvium': 60,
        'fill': 85,
        'landslide_deposit': 90,
        'unknown': 50
    }

    # Risk level thresholds
    THRESHOLDS = {
        RiskLevel.GREEN: (0, 25),
        RiskLevel.YELLOW: (26, 50),
        RiskLevel.ORANGE: (51, 75),
        RiskLevel.RED: (76, 100)
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_coherence: float = 0.3
    ):
        """
        Initialize risk calculator.

        Args:
            weights: Custom weights for risk factors (must sum to 1.0)
            min_coherence: Minimum InSAR coherence for reliable results
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.min_coherence = min_coherence

        # Validate weights
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

    def calculate(self, data: SegmentData) -> RiskResult:
        """
        Calculate risk score for a segment.

        Args:
            data: SegmentData with all input parameters

        Returns:
            RiskResult with score, level, and component scores
        """
        timestamp = datetime.utcnow()

        # Check data quality
        if data.coherence < self.min_coherence:
            return RiskResult(
                segment_id=data.segment_id,
                lat=data.lat,
                lon=data.lon,
                score=0,
                level=RiskLevel.UNKNOWN,
                component_scores={},
                timestamp=timestamp,
                coherence=data.coherence,
                message=f"コヒーレンス低下 ({data.coherence:.2f}) - 信頼性が低い",
                is_reliable=False
            )

        # Calculate component scores
        scores = {}

        scores['deformation_rate'] = self._score_deformation_rate(
            data.deformation_rate_mm_year
        )

        scores['deformation_acceleration'] = self._score_acceleration(
            data.deformation_acceleration
        )

        scores['slope_angle'] = self._score_slope(
            data.slope_angle_degrees
        )

        scores['rainfall_factor'] = self._score_rainfall(
            data.rainfall_48h_mm,
            data.rainfall_7d_mm
        )

        scores['geology_factor'] = self._score_geology(
            data.geology_type
        )

        # Calculate weighted sum
        total_score = sum(
            scores[factor] * self.weights[factor]
            for factor in self.weights.keys()
        )

        # Clamp to 0-100
        total_score = int(min(100, max(0, total_score)))

        # Determine level
        level = self._get_level(total_score)

        # Generate message and insights
        message = self._generate_message(data, scores, total_score, level)
        insights = self._generate_insights(data)

        return RiskResult(
            segment_id=data.segment_id,
            lat=data.lat,
            lon=data.lon,
            score=total_score,
            level=level,
            component_scores=scores,
            timestamp=timestamp,
            coherence=data.coherence,
            message=message,
            is_reliable=True,
            insights=insights
        )

    def _score_deformation_rate(self, rate_mm_year: float) -> float:
        """
        Score deformation rate.

        Uses absolute value since both subsidence and uplift can indicate instability.
        More movement = higher score.
        """
        abs_rate = abs(rate_mm_year)

        if abs_rate < 2:
            return abs_rate * 5  # 0-10
        elif abs_rate < 5:
            return 10 + (abs_rate - 2) * 6.67  # 10-30
        elif abs_rate < 10:
            return 30 + (abs_rate - 5) * 6  # 30-60
        elif abs_rate < 20:
            return 60 + (abs_rate - 10) * 2  # 60-80
        elif abs_rate < 30:
            return 80 + (abs_rate - 20) * 1  # 80-90
        else:
            return min(100, 90 + (abs_rate - 30) * 0.5)  # 90-100

    def _score_acceleration(self, accel: float) -> float:
        """
        Score deformation acceleration.

        Positive acceleration (speeding up) is concerning.
        Negative acceleration (slowing down) is less concerning.
        """
        if accel <= 0:
            return max(0, 20 + accel * 5)  # Slowing = lower score

        # Accelerating deformation
        if accel < 1:
            return 20 + accel * 20  # 20-40
        elif accel < 3:
            return 40 + (accel - 1) * 15  # 40-70
        elif accel < 5:
            return 70 + (accel - 3) * 10  # 70-90
        else:
            return min(100, 90 + (accel - 5) * 2)  # 90-100

    def _score_slope(self, angle_degrees: float) -> float:
        """
        Score slope angle.

        Steeper slopes are more prone to failure.
        Critical angle depends on material, but generally:
        - < 15°: Low risk
        - 15-30°: Moderate risk
        - 30-45°: High risk
        - > 45°: Very high risk
        """
        if angle_degrees < 10:
            return angle_degrees * 1  # 0-10
        elif angle_degrees < 20:
            return 10 + (angle_degrees - 10) * 2  # 10-30
        elif angle_degrees < 30:
            return 30 + (angle_degrees - 20) * 2  # 30-50
        elif angle_degrees < 40:
            return 50 + (angle_degrees - 30) * 3  # 50-80
        elif angle_degrees < 50:
            return 80 + (angle_degrees - 40) * 1.5  # 80-95
        else:
            return min(100, 95 + (angle_degrees - 50) * 0.5)  # 95-100

    def _score_rainfall(
        self,
        rainfall_48h: float,
        rainfall_7d: float = 0.0
    ) -> float:
        """
        Score rainfall factor.

        Both short-term (48h) and longer-term (7d) rainfall contribute.
        Heavy recent rainfall significantly increases slope failure risk.
        """
        # 48-hour rainfall (primary factor)
        if rainfall_48h < 10:
            score_48h = rainfall_48h * 1  # 0-10
        elif rainfall_48h < 30:
            score_48h = 10 + (rainfall_48h - 10) * 1.5  # 10-40
        elif rainfall_48h < 80:
            score_48h = 40 + (rainfall_48h - 30) * 0.8  # 40-80
        elif rainfall_48h < 150:
            score_48h = 80 + (rainfall_48h - 80) * 0.2  # 80-94
        else:
            score_48h = min(100, 94 + (rainfall_48h - 150) * 0.04)  # 94-100

        # 7-day rainfall adds to saturation risk
        if rainfall_7d > 100:
            saturation_bonus = min(20, (rainfall_7d - 100) * 0.1)
            score_48h = min(100, score_48h + saturation_bonus)

        return score_48h

    def _score_geology(self, geology_type: str) -> float:
        """
        Score geology factor.

        Different rock/soil types have inherently different stability.
        """
        return self.GEOLOGY_RISK.get(geology_type.lower(), 50)

    def _get_level(self, score: int) -> RiskLevel:
        """Convert score to risk level."""
        for level, (min_val, max_val) in self.THRESHOLDS.items():
            if min_val <= score <= max_val:
                return level
        return RiskLevel.UNKNOWN

    def _generate_message(
        self,
        data: SegmentData,
        scores: Dict[str, float],
        total_score: int,
        level: RiskLevel
    ) -> str:
        """Generate human-readable message in Japanese."""
        messages = []

        # Main message based on level (Japanese)
        if level == RiskLevel.RED:
            messages.append("【危険】緊急対応が必要です")
        elif level == RiskLevel.ORANGE:
            messages.append("【警戒】現地点検を推奨します")
        elif level == RiskLevel.YELLOW:
            messages.append("【注意】監視強化を推奨します")
        else:
            messages.append("【正常】通常監視を継続")

        # Specific warnings (Japanese) — factual observations only
        if abs(data.deformation_rate_mm_year) > 20:
            messages.append("⚠ 顕著な地盤変動を検出")

        if data.deformation_acceleration > 2:
            messages.append("⚠ 変動が加速中")

        if data.rainfall_48h_mm > 100:
            messages.append("⚠ 大雨警戒")

        return " | ".join(messages)

    def _generate_insights(self, data: SegmentData) -> List[Dict]:
        """Generate data-driven insights for display (without revealing logic)."""
        insights = []

        # Ground movement insight (from InSAR)
        abs_rate = abs(data.deformation_rate_mm_year)
        if abs_rate > 30:
            insights.append({
                "icon": "📡",
                "label": "衛星観測",
                "value": "大きな地盤変動を検出",
                "detail": f"年間 {abs_rate:.0f}mm の動き",
                "severity": "high"
            })
        elif abs_rate > 10:
            insights.append({
                "icon": "📡",
                "label": "衛星観測",
                "value": "地盤変動を検出",
                "detail": f"年間 {abs_rate:.0f}mm の動き",
                "severity": "medium"
            })
        else:
            insights.append({
                "icon": "📡",
                "label": "衛星観測",
                "value": "地盤は安定",
                "detail": "有意な変動なし",
                "severity": "low"
            })

        # Slope insight (from DEM)
        if data.slope_angle_degrees > 35:
            insights.append({
                "icon": "⛰️",
                "label": "地形",
                "value": "急傾斜地",
                "detail": f"傾斜角 {data.slope_angle_degrees:.0f}°",
                "severity": "high"
            })
        elif data.slope_angle_degrees > 20:
            insights.append({
                "icon": "⛰️",
                "label": "地形",
                "value": "やや急な傾斜",
                "detail": f"傾斜角 {data.slope_angle_degrees:.0f}°",
                "severity": "medium"
            })
        else:
            insights.append({
                "icon": "⛰️",
                "label": "地形",
                "value": "緩やかな傾斜",
                "detail": f"傾斜角 {data.slope_angle_degrees:.0f}°",
                "severity": "low"
            })

        # Rainfall insight (JMA AMeDAS)
        rain = data.rainfall_48h_mm  # Actually 24h from AMeDAS
        if rain > 80:
            insights.append({
                "icon": "🌧️", "label": "気象",
                "value": "大雨の影響あり",
                "detail": f"24時間 {rain:.0f}mm",
                "severity": "high"
            })
        elif rain > 30:
            insights.append({
                "icon": "🌧️", "label": "気象",
                "value": "降雨の影響あり",
                "detail": f"24時間 {rain:.0f}mm",
                "severity": "medium"
            })
        elif rain > 0:
            insights.append({
                "icon": "🌧️", "label": "気象",
                "value": "小雨",
                "detail": f"24時間 {rain:.0f}mm",
                "severity": "low"
            })
        else:
            insights.append({
                "icon": "☀️", "label": "気象",
                "value": "降雨なし",
                "detail": "24時間 0mm",
                "severity": "low"
            })

        # Geology insight (from GSI)
        geo_name = GEOLOGY_NAMES_JA.get(data.geology_type, data.geology_type)
        geo_risk = self.GEOLOGY_RISK.get(data.geology_type, 50)
        if geo_risk > 60:
            severity = "high"
        elif geo_risk > 40:
            severity = "medium"
        else:
            severity = "low"
        insights.append({
            "icon": "🪨",
            "label": "地質",
            "value": geo_name,
            "detail": "地質調査データより",
            "severity": severity
        })

        return insights

    def calculate_batch(
        self,
        segments: List[SegmentData]
    ) -> List[RiskResult]:
        """
        Calculate risk scores for multiple segments.

        Args:
            segments: List of SegmentData objects

        Returns:
            List of RiskResult objects
        """
        results = []
        for segment in segments:
            result = self.calculate(segment)
            results.append(result)
        return results

    def get_high_risk_segments(
        self,
        results: List[RiskResult],
        min_level: RiskLevel = RiskLevel.ORANGE
    ) -> List[RiskResult]:
        """
        Filter results to high-risk segments.

        Args:
            results: List of RiskResult objects
            min_level: Minimum risk level to include

        Returns:
            Filtered list of high-risk segments
        """
        level_order = {
            RiskLevel.GREEN: 0,
            RiskLevel.YELLOW: 1,
            RiskLevel.ORANGE: 2,
            RiskLevel.RED: 3,
            RiskLevel.UNKNOWN: -1
        }

        min_order = level_order.get(min_level, 0)

        return [
            r for r in results
            if level_order.get(r.level, -1) >= min_order
        ]


def main():
    """Test risk calculator with sample data"""
    calculator = RiskScoreCalculator()

    # Test cases
    test_segments = [
        SegmentData(
            segment_id="CHUO_001",
            lat=35.70,
            lon=138.60,
            deformation_rate_mm_year=-5.0,
            slope_angle_degrees=25,
            rainfall_48h_mm=30,
            geology_type="sandstone",
            coherence=0.8
        ),
        SegmentData(
            segment_id="CHUO_002",
            lat=35.72,
            lon=138.62,
            deformation_rate_mm_year=-18.0,
            deformation_acceleration=2.5,
            slope_angle_degrees=38,
            rainfall_48h_mm=120,
            geology_type="volcanic_ash",
            coherence=0.7
        ),
        SegmentData(
            segment_id="CHUO_003",
            lat=35.74,
            lon=138.64,
            deformation_rate_mm_year=-2.0,
            slope_angle_degrees=15,
            rainfall_48h_mm=5,
            geology_type="granite",
            coherence=0.9
        ),
    ]

    print("=" * 60)
    print("HighwayLens Risk Calculator - Test Results")
    print("=" * 60)

    for segment in test_segments:
        result = calculator.calculate(segment)

        print(f"\nSegment: {result.segment_id}")
        print(f"  Location: ({result.lat:.4f}, {result.lon:.4f})")
        print(f"  Risk Score: {result.score}")
        print(f"  Level: {result.level.value.upper()}")
        print(f"  Coherence: {result.coherence:.2f}")
        print(f"  Message: {result.message}")
        print(f"  Component scores:")
        for factor, score in result.component_scores.items():
            weight = calculator.weights[factor]
            contribution = score * weight
            print(f"    {factor}: {score:.1f} (weight: {weight}, contrib: {contribution:.1f})")


if __name__ == "__main__":
    main()
