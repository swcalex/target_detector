"""algorithm_spec.md Section 3.4: Convergence Judgment (수렴 판독/판정 모듈)

상태 없는 순수 함수로 판독 연산만 제공한다.
종료 통제(반복/멈춤)는 train_dataset.py의 학습 루프가 담당한다.

표기 규칙:
- k는 0-based 블록 인덱스(구현 내부), 레코드 출력 시 1-based(k+1)로 변환한다.
"""

import numpy as np

THETA_R = 0.05   # 기준 1 상대 기울기 문턱값 (θ₁)
THETA_PI = 0.10  # 기준 2 상대 진폭 문턱값 (θ₂)
EPS = 1e-9       # 부동소수 비교 안정화용 마진 (문턱값 경계 비교)


def block_stats(losses):
    """분석 창(블록)의 통계량을 산출한다.

    Args:
        losses: 블록(직전 BLOCK_EPOCHS epoch)의 epoch별 학습 Loss 값 리스트

    Returns:
        dict: m_first(전반 절반 평균), m_second(후반 절반 평균),
              m(전체 평균), std(σ), max, min
    """
    arr = np.asarray(losses, dtype=np.float64)
    half = len(arr) // 2
    return {
        "m_first": float(arr[:half].mean()),
        "m_second": float(arr[half:].mean()),
        "m": float(arr.mean()),
        "std": float(arr.std()),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def judge_block(k, losses):
    """블록 1건을 판독하여 판독 레코드 dict를 산출한다.

    Args:
        k: 0-based 블록 인덱스 (방금 완료한 블록)
        losses: 해당 블록의 epoch별 학습 Loss 값 리스트

    Returns:
        dict: r, c1, rising_event, pi, c2, dip_threshold,
              dips[(전역 epoch 번호, Loss 값)], stop, epoch_range
    """
    s = block_stats(losses)
    n = len(losses)
    m_first = s["m_first"]

    # 방어: 실제 BCE 학습에서 m_first=0은 발생하지 않으나 분모 0 오류 방지
    if m_first == 0.0:
        r = 0.0
    else:
        r = (m_first - s["m_second"]) / m_first

    c1 = abs(r) <= THETA_R + EPS
    rising_event = r < -THETA_R - EPS

    # 방어: 전 구간이 0인 창에서 Π의 분모(m)가 0이 되는 경우를 방지
    if s["m"] == 0.0:
        pi = 0.0
    else:
        pi = (s["max"] - s["min"]) / s["m"]
    c2 = pi <= THETA_PI + EPS

    dip_threshold = s["m"] - 3.0 * s["std"]
    offset = k * n
    dips = [
        (offset + i + 1, float(loss))
        for i, loss in enumerate(losses)
        if loss < dip_threshold
    ]

    return {
        "k": k,
        "r": r,
        "c1": c1,
        "rising_event": rising_event,
        "pi": pi,
        "c2": c2,
        "dip_threshold": dip_threshold,
        "dips": dips,
        "stop": c1 and c2,
        "epoch_range": (offset + 1, offset + n),
    }


def format_record(record):
    """판독 레코드를 사람이 읽는 텍스트 1줄로 직렬화한다."""
    k = record["k"]
    first, last = record["epoch_range"]
    c1 = "만족" if record["c1"] else "불만족"
    c2 = "만족" if record["c2"] else "불만족"
    rising = "발생" if record["rising_event"] else "없음"
    if record["dips"]:
        dip_desc = ", ".join(f"(epoch {e}, {v:.4f})" for e, v in record["dips"])
        c3 = f"발생 {dip_desc}"
    else:
        c3 = "없음"
    verdict = "학습 종료" if record["stop"] else "계속 학습"

    return (
        f"[block {k + 1} | epoch {first}-{last}] "
        f"r={record['r']:+.4f}  C1: {c1}  "
        f"Π={record['pi']:.4f}  C2: {c2}  "
        f"C3: {c3}  상승 이탈: {rising}  → {verdict}"
    )
